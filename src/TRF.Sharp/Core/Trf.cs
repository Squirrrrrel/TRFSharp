using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp.Core;

/// <summary>
/// C# port of <c>scipy.optimize._lsq.trf</c> — the Trust Region Reflective
/// algorithm for bound-constrained nonlinear least squares
/// (Branch, Coleman &amp; Li, SIAM J. Sci. Comput. 21 (1999) 1–23).
/// </summary>
internal static class Trf
{
    public static LeastSquaresResult Solve(
        Func<Vector<double>, Vector<double>> fun,
        Func<Vector<double>, Matrix<double>> jac,
        Vector<double> x0,
        Vector<double> f0,
        Matrix<double> J0,
        Vector<double> lb,
        Vector<double> ub,
        double ftol,
        double xtol,
        double gtol,
        int? maxNfev,
        XScaleSpec xScale,
        Loss? loss,
        double fScale,
        TrustRegionSolver trSolver,
        TrfOptions trOptions,
        int verbose,
        Func<LeastSquaresResult, bool>? callback)
    {
        // For efficiency it makes sense to run the simplified version of the
        // algorithm when no bounds are imposed (SciPy keeps two functions).
        bool unbounded = true;
        for (int i = 0; i < lb.Count; i++)
        {
            if (lb[i] != double.NegativeInfinity || ub[i] != double.PositiveInfinity)
            {
                unbounded = false;
                break;
            }
        }

        return unbounded
            ? TrfNoBounds(fun, jac, x0, f0, J0, ftol, xtol, gtol, maxNfev, xScale, loss,
                          fScale, trSolver, trOptions, verbose, callback)
            : TrfBounds(fun, jac, x0, f0, J0, lb, ub, ftol, xtol, gtol, maxNfev, xScale, loss,
                        fScale, trSolver, trOptions, verbose, callback);
    }

    // ------------------------------------------------------------------
    // select_step
    // ------------------------------------------------------------------

    /// <summary>
    /// Port of <c>trf.select_step</c>: chooses the best of the constrained
    /// trust-region step, the reflected step and the constrained Cauchy step.
    /// </summary>
    private static (Vector<double> step, Vector<double> stepH, double predictedReduction)
        SelectStep(
            Vector<double> x, ILinearOperator jH, Vector<double> diagH, Vector<double> gH,
            Vector<double> p, Vector<double> pH, Vector<double> d, double delta,
            Vector<double> lb, Vector<double> ub, double theta)
    {
        if (Common.InBounds(x + p, lb, ub))
        {
            double interiorValue = Common.EvaluateQuadratic(jH, gH, pH, diagH);
            return (p, pH, -interiorValue);
        }

        var (pStride, hits) = Common.StepSizeToBound(x, p, lb, ub);

        // Compute the reflected direction.
        var rH = pH.Clone();
        for (int i = 0; i < rH.Count; i++)
        {
            if (hits[i] != 0)
            {
                rH[i] *= -1;
            }
        }
        var r = d.PointwiseMultiply(rH);

        // Restrict the trust-region step so that it hits the bound.
        p = p * pStride;
        pH = pH * pStride;
        var xOnBound = x + p;

        // The reflected direction crosses either the feasible region or the
        // trust-region boundary first.
        var (_, toTr) = Common.IntersectTrustRegion(pH, rH, delta);
        var (toBound, _) = Common.StepSizeToBound(xOnBound, r, lb, ub);

        double rStride = Math.Min(toBound, toTr);
        double rStrideL;
        double rStrideU;
        if (rStride > 0)
        {
            rStrideL = (1 - theta) * pStride / rStride;
            rStrideU = rStride == toBound ? theta * toBound : toTr;
        }
        else
        {
            rStrideL = 0;
            rStrideU = -1;
        }

        double rValue;
        if (rStrideL <= rStrideU)
        {
            var qr = Common.BuildQuadratic1D(jH, gH, rH, pH, diagH);
            var (t, y) = Common.MinimizeQuadratic1D(qr.A, qr.B, rStrideL, rStrideU, qr.C);
            rH = (rH * t) + pH;
            r = d.PointwiseMultiply(rH);
            rValue = y;
        }
        else
        {
            rValue = double.PositiveInfinity;
        }

        // Now correct p_h to make it strictly interior.
        p = p * theta;
        pH = pH * theta;
        double pValue = Common.EvaluateQuadratic(jH, gH, pH, diagH);

        var agH = -gH;
        var ag = d.PointwiseMultiply(agH);

        double agToTr = delta / agH.L2Norm();
        var (agToBound, _) = Common.StepSizeToBound(x, ag, lb, ub);
        double agStride = agToBound < agToTr ? theta * agToBound : agToTr;

        var qa = Common.BuildQuadratic1D(jH, gH, agH, null, diagH);
        var (agT, agValue) = Common.MinimizeQuadratic1D(qa.A, qa.B, 0, agStride);
        agH = agH * agT;
        ag = ag * agT;

        if (pValue < rValue && pValue < agValue)
        {
            return (p, pH, -pValue);
        }
        if (rValue < pValue && rValue < agValue)
        {
            return (r, rH, -rValue);
        }
        return (ag, agH, -agValue);
    }

    // ------------------------------------------------------------------
    // trf_bounds
    // ------------------------------------------------------------------

    private static LeastSquaresResult TrfBounds(
        Func<Vector<double>, Vector<double>> fun,
        Func<Vector<double>, Matrix<double>> jac,
        Vector<double> x0,
        Vector<double> f0,
        Matrix<double> J0,
        Vector<double> lb,
        Vector<double> ub,
        double ftol,
        double xtol,
        double gtol,
        int? maxNfev,
        XScaleSpec xScale,
        Loss? loss,
        double fScale,
        TrustRegionSolver trSolver,
        TrfOptions trOptions,
        int verbose,
        Func<LeastSquaresResult, bool>? callback)
    {
        var x = x0.Clone();
        var f = f0.Clone();
        var fTrue = f0.Clone();
        int nfev = 1;

        var J = J0.Clone();
        int njev = 1;
        int m = J.RowCount;
        int n = J.ColumnCount;

        double cost;
        if (loss is not null)
        {
            var rho = LossRuntime.Rho(loss, fScale, f);
            cost = 0.5 * rho.Row(0).Sum();
            (J, f) = Common.ScaleForRobustLossFunction(J, f, rho);
        }
        else
        {
            cost = 0.5 * f.DotProduct(f);
        }

        var g = Common.ComputeGrad(J, f);

        Vector<double> scale;
        Vector<double> scaleInv;
        if (xScale.IsJac)
        {
            (scale, scaleInv) = Common.ComputeJacScale(J, null);
        }
        else
        {
            scale = xScale.Values.Clone();
            scaleInv = scale.Map(v => 1.0 / v);
        }

        var (v, dv) = Common.ClScalingVector(x, g, lb, ub);
        ApplyScaleInv(v, dv, scaleInv);

        double delta = x0.PointwiseMultiply(scaleInv).PointwiseDivide(v.Map(Math.Sqrt)).L2Norm();
        if (delta == 0)
        {
            delta = 1.0;
        }

        double gNorm = g.PointwiseMultiply(v).InfinityNorm();

        var fAugmented = Vector<double>.Build.Dense(m + n);
        Matrix<double>? jAugmented = trSolver == TrustRegionSolver.Exact
            ? Matrix<double>.Build.Dense(m + n, n)
            : null;

        double regTerm = 0.0;
        bool regularize = trOptions.Regularize;

        int maxNfevValue = maxNfev ?? n * 100;
        double alpha = 0.0; // "Levenberg-Marquardt" parameter

        int? terminationStatus = null;
        int iteration = 0;
        double? stepNorm = null;
        double? actualReduction = null;
        bool callbackStop = false;

        if (verbose == 2)
        {
            Common.PrintHeaderNonlinear();
        }

        while (true)
        {
            (v, dv) = Common.ClScalingVector(x, g, lb, ub);

            gNorm = g.PointwiseMultiply(v).InfinityNorm();
            if (gNorm < gtol)
            {
                terminationStatus = 1;
            }

            if (verbose == 2)
            {
                Common.PrintIterationNonlinear(iteration, nfev, cost, actualReduction, stepNorm, gNorm);
            }

            if (terminationStatus is not null || nfev == maxNfevValue)
            {
                break;
            }

            // Compute variables in "hat" space, accounting for the scaling
            // introduced by `x_scale` (see the module docstring of SciPy's trf.py).
            ApplyScaleInv(v, dv, scaleInv);

            var d = v.Map(Math.Sqrt).PointwiseMultiply(scale);

            // C = diag(g * scale) Jv
            var diagH = g.PointwiseMultiply(dv).PointwiseMultiply(scale);

            var gH = d.PointwiseMultiply(g);

            for (int i = 0; i < m; i++)
            {
                fAugmented[i] = f[i];
            }

            ILinearOperator jH;
            Vector<double>? s = null;
            Matrix<double>? V = null;
            Vector<double>? uf = null;

            Matrix<double>? S = null;
            Matrix<double>? BS = null;
            Vector<double>? gS = null;

            if (trSolver == TrustRegionSolver.Exact)
            {
                var jScaled = Common.RightMultiply(J, d);
                jAugmented!.SetSubMatrix(0, 0, jScaled);
                for (int i = 0; i < n; i++)
                {
                    jAugmented[m + i, i] = Math.Sqrt(Math.Max(0.0, diagH[i]));
                }

                (s, V) = EconomySvd(jAugmented);

                // suf == Vᵀ·g_h == s .* (Uᵀ·f_augmented).  Going through V
                // instead of U is cheaper and far more accurate on
                // ill-conditioned augmented Jacobians — see SolveLsqTrustRegion.
                uf = V.TransposeThisAndMultiply(gH);
                jH = new DenseLinearOperator(jScaled);
            }
            else
            {
                jH = new RightMultipliedOperator(new DenseLinearOperator(J), d);

                if (regularize)
                {
                    var q = Common.BuildQuadratic1D(jH, gH, -gH, null, diagH);
                    double toTr = delta / gH.L2Norm();
                    double agValue = Common.MinimizeQuadratic1D(q.A, q.B, 0, toTr).y;
                    regTerm = -agValue / (delta * delta);
                }

                var lsmrOp = new RegularizedLsqOperator(
                    jH, diagH.Map(t => Math.Sqrt(Math.Max(0.0, t + regTerm))));
                var gnH = Lsmr.Solve(
                    lsmrOp, fAugmented,
                    damp: trOptions.Damp,
                    atol: trOptions.Atol,
                    btol: trOptions.Btol,
                    conlim: trOptions.Conlim,
                    maxiter: trOptions.MaxIter).X;

                S = OrthonormalBasis(gH, gnH);
                var jS = jH.MatMat(S);
                BS = jS.TransposeThisAndMultiply(jS);
                for (int i = 0; i < 2; i++)
                {
                    for (int k = 0; k < 2; k++)
                    {
                        double sum = 0.0;
                        for (int t = 0; t < n; t++)
                        {
                            sum += S[t, i] * diagH[t] * S[t, k];
                        }
                        BS[i, k] += sum;
                    }
                }
                gS = S.TransposeThisAndMultiply(gH);
            }

            // theta controls how far we step back from the bounds.
            double theta = Math.Max(0.995, 1 - gNorm);

            var xNew = x;
            var fNew = f;
            double costNew = cost;
            double actualReductionValue = -1;
            while (actualReductionValue <= 0 && nfev < maxNfevValue)
            {
                Vector<double> pH;
                if (trSolver == TrustRegionSolver.Exact)
                {
                    (pH, alpha, _) = Common.SolveLsqTrustRegion(
                        n, m, uf!, s!, V!, delta, initialAlpha: alpha);
                }
                else
                {
                    var (pS, _) = Common.SolveTrustRegion2D(BS!, gS!, delta);
                    pH = S!.Multiply(pS);
                }

                var p = d.PointwiseMultiply(pH); // Trust-region solution in the original space.
                var (step, stepH, predictedReduction) = SelectStep(
                    x, jH, diagH, gH, p, pH, d, delta, lb, ub, theta);

                xNew = Common.MakeStrictlyFeasible(x + step, lb, ub, 0.0);
                fNew = fun(xNew);
                nfev += 1;

                double stepHNorm = stepH.L2Norm();

                if (!Common.AllFinite(fNew))
                {
                    delta = 0.25 * stepHNorm;
                    continue;
                }

                costNew = loss is not null
                    ? LossRuntime.CostOnly(loss, fScale, fNew)
                    : 0.5 * fNew.DotProduct(fNew);
                actualReductionValue = cost - costNew;

                var (deltaNew, ratio) = Common.UpdateTrRadius(
                    delta, actualReductionValue, predictedReduction,
                    stepHNorm, stepHNorm > 0.95 * delta);

                stepNorm = step.L2Norm();
                terminationStatus = Common.CheckTermination(
                    actualReductionValue, cost, stepNorm.Value, x.L2Norm(), ratio, ftol, xtol);
                if (terminationStatus is not null)
                {
                    break;
                }

                alpha *= delta / deltaNew;
                delta = deltaNew;
            }

            if (actualReductionValue > 0)
            {
                x = xNew;
                f = fNew;
                fTrue = fNew.Clone();
                cost = costNew;

                J = jac(x);
                njev += 1;

                if (loss is not null)
                {
                    var rho = LossRuntime.Rho(loss, fScale, f);
                    (J, f) = Common.ScaleForRobustLossFunction(J, f, rho);
                }

                g = Common.ComputeGrad(J, f);

                if (xScale.IsJac)
                {
                    (scale, scaleInv) = Common.ComputeJacScale(J, scaleInv);
                }
            }
            else
            {
                stepNorm = 0;
                actualReductionValue = 0;
            }

            actualReduction = actualReductionValue;
            iteration += 1;

            if (callback is not null)
            {
                var intermediate = new LeastSquaresResult
                {
                    X = x,
                    Fun = fTrue,
                    Nit = iteration,
                    Nfev = nfev,
                    Cost = cost,
                    Optimality = gNorm,
                    Status = terminationStatus ?? 0,
                };
                if (callback(intermediate))
                {
                    terminationStatus = -2;
                    callbackStop = true;
                    break;
                }
            }
        }

        int status = terminationStatus ?? 0;
        var activeMask = Common.FindActiveConstraints(x, lb, ub, xtol);
        return new LeastSquaresResult
        {
            X = x,
            Cost = cost,
            Fun = fTrue,
            Jac = J,
            Grad = g,
            Optimality = gNorm,
            ActiveMask = activeMask,
            Nfev = nfev,
            Njev = njev,
            Nit = iteration,
            Status = status,
            Message = TerminationMessages.Get(status, callbackStop),
            Success = status > 0,
        };
    }

    // ------------------------------------------------------------------
    // trf_no_bounds
    // ------------------------------------------------------------------

    private static LeastSquaresResult TrfNoBounds(
        Func<Vector<double>, Vector<double>> fun,
        Func<Vector<double>, Matrix<double>> jac,
        Vector<double> x0,
        Vector<double> f0,
        Matrix<double> J0,
        double ftol,
        double xtol,
        double gtol,
        int? maxNfev,
        XScaleSpec xScale,
        Loss? loss,
        double fScale,
        TrustRegionSolver trSolver,
        TrfOptions trOptions,
        int verbose,
        Func<LeastSquaresResult, bool>? callback)
    {
        var x = x0.Clone();
        var f = f0.Clone();
        var fTrue = f0.Clone();
        int nfev = 1;

        var J = J0.Clone();
        int njev = 1;
        int m = J.RowCount;
        int n = J.ColumnCount;

        double cost;
        if (loss is not null)
        {
            var rho = LossRuntime.Rho(loss, fScale, f);
            cost = 0.5 * rho.Row(0).Sum();
            (J, f) = Common.ScaleForRobustLossFunction(J, f, rho);
        }
        else
        {
            cost = 0.5 * f.DotProduct(f);
        }

        var g = Common.ComputeGrad(J, f);

        Vector<double> scale;
        Vector<double> scaleInv;
        if (xScale.IsJac)
        {
            (scale, scaleInv) = Common.ComputeJacScale(J, null);
        }
        else
        {
            scale = xScale.Values.Clone();
            scaleInv = scale.Map(v => 1.0 / v);
        }

        double delta = x0.PointwiseMultiply(scaleInv).L2Norm();
        if (delta == 0)
        {
            delta = 1.0;
        }

        double regTerm = 0.0;
        bool regularize = trOptions.Regularize;

        int maxNfevValue = maxNfev ?? n * 100;
        double alpha = 0.0;

        int? terminationStatus = null;
        int iteration = 0;
        double? stepNorm = null;
        double? actualReduction = null;
        double gNorm = g.InfinityNorm();
        bool callbackStop = false;

        if (verbose == 2)
        {
            Common.PrintHeaderNonlinear();
        }

        while (true)
        {
            gNorm = g.InfinityNorm();
            if (gNorm < gtol)
            {
                terminationStatus = 1;
            }

            if (verbose == 2)
            {
                Common.PrintIterationNonlinear(iteration, nfev, cost, actualReduction, stepNorm, gNorm);
            }

            if (terminationStatus is not null || nfev == maxNfevValue)
            {
                break;
            }

            var d = scale;
            var gH = d.PointwiseMultiply(g);

            ILinearOperator jH;
            Vector<double>? s = null;
            Matrix<double>? V = null;
            Vector<double>? uf = null;
            Matrix<double>? S = null;
            Matrix<double>? BS = null;
            Vector<double>? gS = null;

            if (trSolver == TrustRegionSolver.Exact)
            {
                var jScaled = Common.RightMultiply(J, d);
                jH = new DenseLinearOperator(jScaled);
                (s, V) = EconomySvd(jScaled);

                // See the trf_bounds branch: Vᵀ·g_h == s .* (Uᵀ·f).
                uf = V.TransposeThisAndMultiply(gH);
            }
            else
            {
                jH = new RightMultipliedOperator(new DenseLinearOperator(J), d);

                if (regularize)
                {
                    var q = Common.BuildQuadratic1D(jH, gH, -gH);
                    double toTr = delta / gH.L2Norm();
                    double agValue = Common.MinimizeQuadratic1D(q.A, q.B, 0, toTr).y;
                    regTerm = -agValue / (delta * delta);
                }

                double dampFull = Math.Sqrt(trOptions.Damp * trOptions.Damp + regTerm);
                var gnH = Lsmr.Solve(
                    jH, f,
                    damp: dampFull,
                    atol: trOptions.Atol,
                    btol: trOptions.Btol,
                    conlim: trOptions.Conlim,
                    maxiter: trOptions.MaxIter).X;

                S = OrthonormalBasis(gH, gnH);
                var jS = jH.MatMat(S);
                BS = jS.TransposeThisAndMultiply(jS);
                gS = S.TransposeThisAndMultiply(gH);
            }

            var xNew = x;
            var fNew = f;
            double costNew = cost;
            double actualReductionValue = -1;
            while (actualReductionValue <= 0 && nfev < maxNfevValue)
            {
                Vector<double> stepH;
                if (trSolver == TrustRegionSolver.Exact)
                {
                    (stepH, alpha, _) = Common.SolveLsqTrustRegion(
                        n, m, uf!, s!, V!, delta, initialAlpha: alpha);
                }
                else
                {
                    var (pS, _) = Common.SolveTrustRegion2D(BS!, gS!, delta);
                    stepH = S!.Multiply(pS);
                }

                double predictedReduction = -Common.EvaluateQuadratic(jH, gH, stepH);
                var step = d.PointwiseMultiply(stepH);
                xNew = x + step;
                fNew = fun(xNew);
                nfev += 1;

                double stepHNorm = stepH.L2Norm();

                if (!Common.AllFinite(fNew))
                {
                    delta = 0.25 * stepHNorm;
                    continue;
                }

                costNew = loss is not null
                    ? LossRuntime.CostOnly(loss, fScale, fNew)
                    : 0.5 * fNew.DotProduct(fNew);
                actualReductionValue = cost - costNew;

                var (deltaNew, ratio) = Common.UpdateTrRadius(
                    delta, actualReductionValue, predictedReduction,
                    stepHNorm, stepHNorm > 0.95 * delta);

                stepNorm = step.L2Norm();
                terminationStatus = Common.CheckTermination(
                    actualReductionValue, cost, stepNorm.Value, x.L2Norm(), ratio, ftol, xtol);
                if (terminationStatus is not null)
                {
                    break;
                }

                alpha *= delta / deltaNew;
                delta = deltaNew;
            }

            if (actualReductionValue > 0)
            {
                x = xNew;
                f = fNew;
                fTrue = fNew.Clone();
                cost = costNew;

                J = jac(x);
                njev += 1;

                if (loss is not null)
                {
                    var rho = LossRuntime.Rho(loss, fScale, f);
                    (J, f) = Common.ScaleForRobustLossFunction(J, f, rho);
                }

                g = Common.ComputeGrad(J, f);

                if (xScale.IsJac)
                {
                    (scale, scaleInv) = Common.ComputeJacScale(J, scaleInv);
                }
            }
            else
            {
                stepNorm = 0;
                actualReductionValue = 0;
            }

            actualReduction = actualReductionValue;
            iteration += 1;

            if (callback is not null)
            {
                var intermediate = new LeastSquaresResult
                {
                    X = x,
                    Fun = fTrue,
                    Nit = iteration,
                    Nfev = nfev,
                    Cost = cost,
                    Optimality = gNorm,
                    Status = terminationStatus ?? 0,
                };
                if (callback(intermediate))
                {
                    terminationStatus = -2;
                    callbackStop = true;
                    break;
                }
            }
        }

        int status = terminationStatus ?? 0;
        return new LeastSquaresResult
        {
            X = x,
            Cost = cost,
            Fun = fTrue,
            Jac = J,
            Grad = g,
            Optimality = gNorm,
            ActiveMask = new int[x.Count],
            Nfev = nfev,
            Njev = njev,
            Nit = iteration,
            Status = status,
            Message = TerminationMessages.Get(status, callbackStop),
            Success = status > 0,
        };
    }

    // ------------------------------------------------------------------
    // helpers
    // ------------------------------------------------------------------

    private static void ApplyScaleInv(Vector<double> v, Vector<double> dv, Vector<double> scaleInv)
    {
        for (int i = 0; i < v.Count; i++)
        {
            if (dv[i] != 0)
            {
                v[i] *= scaleInv[i];
            }
        }
    }

    /// <summary>
    /// Singular values and right singular vectors of <c>A</c>: A = U diag(s) Vᵀ
    /// with V (n×k), k = min(m, n). The left factor U is <b>never formed</b>.
    /// </summary>
    /// <remarks>
    /// Only <c>s</c> and <c>V</c> are ever consumed: the trust-region
    /// subproblem is driven by <c>suf = Vᵀ·g_h</c>, and the left factor enters
    /// nowhere. For tall matrices — which is the normal case, since the
    /// augmented Jacobian has m+n rows for n parameters — that allows the
    /// factorisation to run through a Householder thin QR that produces only
    /// the n×n factor R, followed by an SVD of R:
    /// <c>A = QR</c>, <c>R = U_R·S·Vᵀ</c> ⇒ <c>s(A) = s(R)</c>, <c>V(A) = V(R)</c>.
    /// MathNet's managed full SVD costs 2 orders of magnitude more and scales
    /// worse than O(m·n²); see <c>bench/csharp/SvdProbe</c> for the A/B
    /// (25× at m = 256, 154× at m = 1024, 563× at m = 4096, n = 9, with the
    /// Gauss-Newton step reproduced to ~1e-13).
    /// </remarks>
    private static (Vector<double> s, Matrix<double> V) EconomySvd(Matrix<double> A)
    {
        // The QR route needs m >= n. It is used whenever the matrix is tall
        // enough that the n×n half dominates anyway; below that the direct SVD
        // is equally fast and keeps full accuracy.
        if (A.RowCount >= 2 * A.ColumnCount)
        {
            var svdR = ThinQr.FactorR(A).Svd();
            return (svdR.S, svdR.VT.Transpose());
        }

        var svd = A.Svd();
        int k = svd.S.Count;
        return (svd.S, svd.VT.SubMatrix(0, k, 0, A.ColumnCount).Transpose());
    }

    /// <summary>
    /// Modified Gram–Schmidt on two vectors — equivalent to SciPy's
    /// <c>qr(np.vstack((g_h, gn_h)).T, mode='economic')</c>.
    /// </summary>
    private static Matrix<double> OrthonormalBasis(Vector<double> a, Vector<double> b)
    {
        int n = a.Count;
        var S = Matrix<double>.Build.Dense(n, 2);

        var c0 = a.Clone();
        double n0 = c0.L2Norm();
        if (n0 == 0)
        {
            c0 = Vector<double>.Build.Dense(n, 0.0);
            c0[0] = 1.0;
        }
        else
        {
            c0 = c0 / n0;
        }
        S.SetColumn(0, c0);

        var c1 = b - (c0 * c0.DotProduct(b));
        double n1 = c1.L2Norm();
        if (n1 <= 1e-12 * Math.Max(1.0, b.L2Norm()))
        {
            // b is numerically parallel to a: use any unit vector orthogonal to a.
            c1 = Vector<double>.Build.Dense(n, 0.0);
            int idx = 0;
            double maxAbs = 0.0;
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(c0[i]) > maxAbs)
                {
                    maxAbs = Math.Abs(c0[i]);
                    idx = i;
                }
            }
            c1[idx == 0 && n > 1 ? 1 : 0] = 1.0;
            c1 = c1 - (c0 * c0.DotProduct(c1));
            double n1b = c1.L2Norm();
            c1 = n1b == 0 ? Vector<double>.Build.Dense(n, 0.0) : c1 / n1b;
        }
        else
        {
            c1 = c1 / n1;
        }
        S.SetColumn(1, c1);

        return S;
    }
}

/// <summary>SciPy's <c>TERMINATION_MESSAGES</c> for <c>least_squares</c>.</summary>
internal static class TerminationMessages
{
    private static readonly Dictionary<int, string> Messages = new()
    {
        [-2] = "Stopped because `callback` function raised `StopIteration` or returned `True`",
        [-1] = "Improper input parameters status returned from `leastsq`",
        [0] = "The maximum number of function evaluations is exceeded.",
        [1] = "`gtol` termination condition is satisfied.",
        [2] = "`ftol` termination condition is satisfied.",
        [3] = "`xtol` termination condition is satisfied.",
        [4] = "Both `ftol` and `xtol` termination conditions are satisfied.",
    };

    public static string Get(int status, bool callbackStop) =>
        callbackStop && status == -2
            ? Messages[-2]
            : Messages.TryGetValue(status, out var message) ? message : string.Empty;
}
