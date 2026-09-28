using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp.Core;

/// <summary>
/// Ports of the shared helpers from <c>scipy.optimize._lsq.common</c>
/// (trust-region subproblem solvers, bound bookkeeping, quadratic forms,
/// termination checks).  Kept numerically identical to the Python original.
/// </summary>
internal static class Common
{
    /// <summary>Machine epsilon for double, as used by SciPy.</summary>
    internal const double Eps = 2.220446049250313e-16;

    // ------------------------------------------------------------------
    // Trust-region subproblem machinery
    // ------------------------------------------------------------------

    /// <summary>
    /// Port of <c>intersect_trust_region</c>. Finds the intersection of the line
    /// x + s·t with the trust-region boundary ‖x + s·t‖ = Delta.
    /// </summary>
    public static (double tNeg, double tPos) IntersectTrustRegion(
        Vector<double> x, Vector<double> s, double delta)
    {
        double a = s.DotProduct(s);
        if (a == 0.0)
        {
            throw new ArgumentException("`s` is zero.");
        }

        double b = x.DotProduct(s);
        double c = x.DotProduct(x) - delta * delta;
        if (c > 0)
        {
            // Defensive deviation from SciPy: `p` is rescaled to have norm
            // exactly Delta immediately before this call, so a violation at
            // rounding level is expected and must not abort the iteration.
            double reference = Math.Max(x.DotProduct(x), delta * delta);
            if (c <= 1e-12 * reference)
            {
                c = 0.0;
            }
            else
            {
                throw new ArgumentException("`x` is not within the trust region.");
            }
        }

        double d = Math.Sqrt(b * b - a * c);
        double q = -(b + Math.CopySign(d, b));
        double t1 = q / a;
        double t2 = c / q;

        return t1 < t2 ? (t1, t2) : (t2, t1);
    }

    private static (double phi, double phiPrime) PhiAndDerivative(
        double alpha, Vector<double> suf, Vector<double> s, double delta)
    {
        var denom = s.Map(v => v * v + alpha);
        double pNorm = suf.PointwiseDivide(denom).L2Norm();
        double phi = pNorm - delta;
        double sum = 0.0;
        for (int i = 0; i < suf.Count; i++)
        {
            double den = denom[i];
            sum += suf[i] * suf[i] / (den * den * den);
        }
        double phiPrime = -sum / pNorm;
        return (phi, phiPrime);
    }

    /// <summary>
    /// Port of <c>solve_lsq_trust_region</c> — the J.J. Moré / MINPACK style
    /// trust-region solver driven by a single SVD of the Jacobian.
    /// </summary>
    /// <summary>
    /// Same as SciPy's <c>solve_lsq_trust_region</c>, except that it takes
    /// <paramref name="suf"/> = <c>s .* (Uᵀ·f)</c> directly.
    /// </summary>
    /// <remarks>
    /// Because the augmented Jacobian rows carry a zero right-hand side,
    /// <c>Aᵀ·f_augmented = d .* g = g_h</c>, and therefore
    /// <c>suf = Vᵀ·g_h</c>.  Computing it that way avoids forming
    /// <c>Uᵀ·f</c> at all: MathNet's left singular vectors lose roughly seven
    /// digits on Jacobians whose smallest singular value is ~1e-10 relative to
    /// the largest (LAPACK, which SciPy uses, does not), which showed up as a
    /// 4× wrong Gauss-Newton step on ill-conditioned bound-constrained
    /// problems.  <c>V</c> (the right singular vectors) is accurate in both.
    /// </remarks>
    public static (Vector<double> p, double alpha, int nIter) SolveLsqTrustRegion(
        int n, int m, Vector<double> suf, Vector<double> s, Matrix<double> V,
        double delta, double? initialAlpha, double rtol = 0.01, int maxIter = 10)
    {
        var uf = suf.PointwiseDivide(s);

        bool fullRank;
        if (m >= n)
        {
            double threshold = Eps * m * s[0];
            fullRank = s[s.Count - 1] > threshold;
        }
        else
        {
            fullRank = false;
        }

        if (fullRank)
        {
            var pGn = -V.Multiply(uf.PointwiseDivide(s));
            if (pGn.L2Norm() <= delta)
            {
                return (pGn, 0.0, 0);
            }
        }

        double alphaUpper = suf.L2Norm() / delta;

        double alphaLower;
        if (fullRank)
        {
            var (phi, phiPrime) = PhiAndDerivative(0.0, suf, s, delta);
            alphaLower = -phi / phiPrime;
        }
        else
        {
            alphaLower = 0.0;
        }

        double alpha;
        if (initialAlpha is null || (!fullRank && initialAlpha.Value == 0.0))
        {
            alpha = Math.Max(0.001 * alphaUpper, Math.Sqrt(alphaLower * alphaUpper));
        }
        else
        {
            alpha = initialAlpha.Value;
        }

        int it;
        for (it = 0; it < maxIter; it++)
        {
            if (alpha < alphaLower || alpha > alphaUpper)
            {
                alpha = Math.Max(0.001 * alphaUpper, Math.Sqrt(alphaLower * alphaUpper));
            }

            var (phi, phiPrime) = PhiAndDerivative(alpha, suf, s, delta);

            if (phi < 0)
            {
                alphaUpper = alpha;
            }

            double ratio = phi / phiPrime;
            alphaLower = Math.Max(alphaLower, alpha - ratio);
            alpha -= (phi + delta) * ratio / delta;

            if (Math.Abs(phi) < rtol * delta)
            {
                break;
            }
        }

        var p = -V.Multiply(suf.PointwiseDivide(s.Map(v => v * v + alpha)));

        // Make the norm of p equal to Delta, p is changed only slightly during
        // this.  It prevents p from lying outside the trust region.
        p = p * (delta / p.L2Norm());

        return (p, alpha, Math.Min(it + 1, maxIter));
    }

    /// <summary>
    /// Port of <c>solve_trust_region_2d</c>: solves a general 2-D trust-region
    /// problem by reducing it to a quartic (as SciPy does with <c>numpy.roots</c>).
    /// </summary>
    public static (Vector<double> p, bool newtonStep) SolveTrustRegion2D(
        Matrix<double> B, Vector<double> g, double delta)
    {
        try
        {
            var chol = B.Cholesky();
            var p = -chol.Solve(g);
            if (p.DotProduct(p) <= delta * delta)
            {
                return (p, true);
            }
        }
        catch (Exception)
        {
            // Not positive definite — fall through to the quartic branch.
        }

        double a = B[0, 0] * delta * delta;
        double b = B[0, 1] * delta * delta;
        double c = B[1, 1] * delta * delta;

        double d = g[0] * delta;
        double f = g[1] * delta;

        // numpy.roots takes coefficients in descending order.
        double[] coeffs =
        {
            -b + d,
            2 * (a - c + f),
            6 * b,
            2 * (-a + c + f),
            -b - d
        };

        var roots = PolynomialRoots.FindRoots(coeffs);
        var ts = new List<double>();
        foreach (var root in roots)
        {
            if (Math.Abs(root.Imaginary) <= 1e-10 * Math.Max(1.0, Math.Abs(root.Real)))
            {
                ts.Add(root.Real);
            }
        }

        if (ts.Count == 0)
        {
            // Degenerate quartic (SciPy would produce NaNs here).  Fall back to
            // the scaled steepest-descent point.
            var fallback = -g * (delta / Math.Max(g.L2Norm(), double.Epsilon));
            return (fallback, false);
        }

        var candidates = Matrix<double>.Build.Dense(2, ts.Count);
        for (int i = 0; i < ts.Count; i++)
        {
            double t = ts[i];
            candidates[0, i] = delta * (2 * t / (1 + t * t));
            candidates[1, i] = delta * ((1 - t * t) / (1 + t * t));
        }

        var bp = B.Multiply(candidates);
        int best = 0;
        double bestValue = double.MaxValue;
        for (int i = 0; i < ts.Count; i++)
        {
            double value = 0.5 * candidates.Column(i).DotProduct(bp.Column(i)) + g.DotProduct(candidates.Column(i));
            if (value < bestValue)
            {
                bestValue = value;
                best = i;
            }
        }

        return (candidates.Column(best), false);
    }

    /// <summary>
    /// Port of <c>update_tr_radius</c>.
    /// </summary>
    public static (double delta, double ratio) UpdateTrRadius(
        double delta, double actualReduction, double predictedReduction,
        double stepNorm, bool boundHit)
    {
        double ratio;
        if (predictedReduction > 0)
        {
            ratio = actualReduction / predictedReduction;
        }
        else if (predictedReduction == 0 && actualReduction == 0)
        {
            ratio = 1;
        }
        else
        {
            ratio = 0;
        }

        if (ratio < 0.25)
        {
            delta = 0.25 * stepNorm;
        }
        else if (ratio > 0.75 && boundHit)
        {
            delta *= 2.0;
        }

        return (delta, ratio);
    }

    // ------------------------------------------------------------------
    // Quadratic forms
    // ------------------------------------------------------------------

    /// <summary>Result of <see cref="BuildQuadratic1D"/>.</summary>
    public readonly struct Quadratic1D
    {
        /// <summary>Coefficient of t².</summary>
        public readonly double A;

        /// <summary>Coefficient of t.</summary>
        public readonly double B;

        /// <summary>Free term (0 when no <c>s0</c> is supplied).</summary>
        public readonly double C;

        public Quadratic1D(double a, double b, double c)
        {
            A = a;
            B = b;
            C = c;
        }
    }

    /// <summary>
    /// Port of <c>build_quadratic_1d</c>.  Parameterizes
    /// 0.5·(s0 + s·t)ᵀ(JᵀJ + diag)(s0 + s·t) + gᵀ(s0 + s·t) along the line.
    /// </summary>
    public static Quadratic1D BuildQuadratic1D(
        ILinearOperator J, Vector<double> g, Vector<double> s,
        Vector<double>? s0 = null, Vector<double>? diag = null)
    {
        var v = J.MatVec(s);
        double a = v.DotProduct(v);
        if (diag is not null)
        {
            a += s.PointwiseMultiply(diag).DotProduct(s);
        }
        a *= 0.5;

        double b = g.DotProduct(s);

        double c = 0.0;
        if (s0 is not null)
        {
            var u = J.MatVec(s0);
            b += u.DotProduct(v);
            c = 0.5 * u.DotProduct(u) + g.DotProduct(s0);
            if (diag is not null)
            {
                b += s0.PointwiseMultiply(diag).DotProduct(s);
                c += 0.5 * s0.PointwiseMultiply(diag).DotProduct(s0);
            }
        }

        return new Quadratic1D(a, b, c);
    }

    /// <summary>
    /// Port of <c>minimize_quadratic_1d</c>: minimize a·t² + b·t + c over [lb, ub].
    /// </summary>
    public static (double t, double y) MinimizeQuadratic1D(
        double a, double b, double lb, double ub, double c = 0.0)
    {
        var candidates = new List<double>(3) { lb, ub };
        if (a != 0)
        {
            double extremum = -0.5 * b / a;
            if (lb < extremum && extremum < ub)
            {
                candidates.Add(extremum);
            }
        }

        double bestT = candidates[0];
        double bestY = candidates[0] * (a * candidates[0] + b) + c;
        foreach (double t in candidates)
        {
            double y = t * (a * t + b) + c;
            if (y < bestY)
            {
                bestY = y;
                bestT = t;
            }
        }

        return (bestT, bestY);
    }

    /// <summary>
    /// Port of <c>evaluate_quadratic</c>: 0.5·sᵀ(JᵀJ + diag)·s + gᵀs.
    /// </summary>
    public static double EvaluateQuadratic(
        ILinearOperator J, Vector<double> g, Vector<double> s, Vector<double>? diag = null)
    {
        var js = J.MatVec(s);
        double q = js.DotProduct(js);
        if (diag is not null)
        {
            q += s.PointwiseMultiply(diag).DotProduct(s);
        }
        return 0.5 * q + s.DotProduct(g);
    }

    // ------------------------------------------------------------------
    // Bound bookkeeping
    // ------------------------------------------------------------------

    /// <summary>Port of <c>in_bounds</c>.</summary>
    public static bool InBounds(Vector<double> x, Vector<double> lb, Vector<double> ub)
    {
        for (int i = 0; i < x.Count; i++)
        {
            if (x[i] < lb[i] || x[i] > ub[i])
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Port of <c>step_size_to_bound</c>.  Returns the smallest t ≥ 0 such that
    /// x + s·t reaches a bound, plus a per-component indicator
    /// (0 = not hit, −1 = lower bound, +1 = upper bound).
    /// </summary>
    public static (double step, int[] hits) StepSizeToBound(
        Vector<double> x, Vector<double> s, Vector<double> lb, Vector<double> ub)
    {
        int n = x.Count;
        var steps = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (s[i] == 0.0)
            {
                steps[i] = double.PositiveInfinity;
                continue;
            }

            double toLower = (lb[i] - x[i]) / s[i];
            double toUpper = (ub[i] - x[i]) / s[i];
            steps[i] = Math.Max(toLower, toUpper);
        }

        double minStep = double.PositiveInfinity;
        for (int i = 0; i < n; i++)
        {
            if (steps[i] < minStep)
            {
                minStep = steps[i];
            }
        }

        var hits = new int[n];
        for (int i = 0; i < n; i++)
        {
            hits[i] = steps[i] == minStep ? Math.Sign(s[i]) : 0;
        }

        return (minStep, hits);
    }

    /// <summary>Port of <c>find_active_constraints</c>.</summary>
    public static int[] FindActiveConstraints(
        Vector<double> x, Vector<double> lb, Vector<double> ub, double rtol)
    {
        int n = x.Count;
        var active = new int[n];

        if (rtol == 0)
        {
            for (int i = 0; i < n; i++)
            {
                if (x[i] <= lb[i])
                {
                    active[i] = -1;
                }
                else if (x[i] >= ub[i])
                {
                    active[i] = 1;
                }
            }
            return active;
        }

        for (int i = 0; i < n; i++)
        {
            double lowerDist = x[i] - lb[i];
            double upperDist = ub[i] - x[i];

            double lowerThreshold = rtol * Math.Max(1.0, Math.Abs(lb[i]));
            double upperThreshold = rtol * Math.Max(1.0, Math.Abs(ub[i]));

            if (double.IsFinite(lb[i]) && lowerDist <= Math.Min(upperDist, lowerThreshold))
            {
                active[i] = -1;
            }
            else if (double.IsFinite(ub[i]) && upperDist <= Math.Min(lowerDist, upperThreshold))
            {
                active[i] = 1;
            }
        }

        return active;
    }

    private static double NextAfter(double from, double to)
    {
        if (to > from)
        {
            return Math.BitIncrement(from);
        }
        if (to < from)
        {
            return Math.BitDecrement(from);
        }
        return from;
    }

    /// <summary>
    /// Port of <c>make_strictly_feasible</c>: shift a point into the interior of
    /// the feasible region.  <paramref name="rstep"/> = 0 uses the next
    /// representable floating point value.
    /// </summary>
    public static Vector<double> MakeStrictlyFeasible(
        Vector<double> x, Vector<double> lb, Vector<double> ub, double rstep = 1e-10)
    {
        var xNew = x.Clone();
        var active = FindActiveConstraints(x, lb, ub, rstep);

        for (int i = 0; i < x.Count; i++)
        {
            if (active[i] == -1)
            {
                xNew[i] = rstep == 0
                    ? NextAfter(lb[i], ub[i])
                    : lb[i] + rstep * Math.Max(1.0, Math.Abs(lb[i]));
            }
            else if (active[i] == 1)
            {
                xNew[i] = rstep == 0
                    ? NextAfter(ub[i], lb[i])
                    : ub[i] - rstep * Math.Max(1.0, Math.Abs(ub[i]));
            }
        }

        for (int i = 0; i < x.Count; i++)
        {
            if (xNew[i] < lb[i] || xNew[i] > ub[i])
            {
                xNew[i] = 0.5 * (lb[i] + ub[i]);
            }
        }

        return xNew;
    }

    /// <summary>
    /// Port of <c>CL_scaling_vector</c> — the Coleman–Li scaling vector and the
    /// diagonal of its Jacobian.
    /// </summary>
    public static (Vector<double> v, Vector<double> dv) ClScalingVector(
        Vector<double> x, Vector<double> g, Vector<double> lb, Vector<double> ub)
    {
        int n = x.Count;
        var v = Vector<double>.Build.Dense(n, 1.0);
        var dv = Vector<double>.Build.Dense(n, 0.0);

        for (int i = 0; i < n; i++)
        {
            if (g[i] < 0 && double.IsFinite(ub[i]))
            {
                v[i] = ub[i] - x[i];
                dv[i] = -1;
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (g[i] > 0 && double.IsFinite(lb[i]))
            {
                v[i] = x[i] - lb[i];
                dv[i] = 1;
            }
        }

        return (v, dv);
    }

    // ------------------------------------------------------------------
    // Simple helpers
    // ------------------------------------------------------------------

    /// <summary>Port of <c>compute_grad</c>: Jᵀ·f.</summary>
    public static Vector<double> ComputeGrad(Matrix<double> J, Vector<double> f)
        => J.TransposeThisAndMultiply(f);

    /// <summary>
    /// Port of <c>compute_jac_scale</c>: column norms of J (monotonically
    /// non-decreasing across iterations when <paramref name="scaleInvOld"/> is
    /// supplied).
    /// </summary>
    public static (Vector<double> scale, Vector<double> scaleInv) ComputeJacScale(
        Matrix<double> J, Vector<double>? scaleInvOld = null)
    {
        var scaleInv = Vector<double>.Build.Dense(J.ColumnCount);
        for (int j = 0; j < J.ColumnCount; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < J.RowCount; i++)
            {
                sum += J[i, j] * J[i, j];
            }
            scaleInv[j] = Math.Sqrt(sum);
        }

        if (scaleInvOld is null)
        {
            scaleInv = scaleInv.Map(t => t == 0 ? 1.0 : t);
        }
        else
        {
            scaleInv = scaleInv.Map2((a, b) => Math.Max(a, b), scaleInvOld);
        }

        return (scaleInv.Map(t => 1.0 / t), scaleInv);
    }

    /// <summary>Port of <c>right_multiply</c> for a dense matrix: J·diag(d).</summary>
    public static Matrix<double> RightMultiply(Matrix<double> J, Vector<double> d)
    {
        var result = J.Clone();
        for (int j = 0; j < result.ColumnCount; j++)
        {
            double dj = d[j];
            for (int i = 0; i < result.RowCount; i++)
            {
                result[i, j] *= dj;
            }
        }
        return result;
    }

    /// <summary>Port of <c>left_multiply</c> for a dense matrix: diag(d)·J.</summary>
    public static Matrix<double> LeftMultiply(Matrix<double> J, Vector<double> d)
    {
        var result = J.Clone();
        for (int i = 0; i < result.RowCount; i++)
        {
            double di = d[i];
            for (int j = 0; j < result.ColumnCount; j++)
            {
                result[i, j] *= di;
            }
        }
        return result;
    }

    /// <summary>Port of <c>check_termination</c>.</summary>
    public static int? CheckTermination(
        double dF, double F, double dxNorm, double xNorm, double ratio,
        double ftol, double xtol)
    {
        bool ftolSatisfied = dF < ftol * F && ratio > 0.25;
        bool xtolSatisfied = dxNorm < xtol * (xtol + xNorm);

        if (ftolSatisfied && xtolSatisfied)
        {
            return 4;
        }
        if (ftolSatisfied)
        {
            return 2;
        }
        if (xtolSatisfied)
        {
            return 3;
        }
        return null;
    }

    /// <summary>
    /// Port of <c>scale_for_robust_loss_function</c>: rescales J and f so that the
    /// Gauss–Newton model matches the Hessian of the robust loss.
    /// </summary>
    public static (Matrix<double> J, Vector<double> f) ScaleForRobustLossFunction(
        Matrix<double> J, Vector<double> f, Matrix<double> rho)
    {
        var jScale = Vector<double>.Build.Dense(f.Count);
        var rho1 = rho.Row(1);
        var rho2 = rho.Row(2);
        for (int i = 0; i < f.Count; i++)
        {
            double value = rho1[i] + 2 * rho2[i] * f[i] * f[i];
            if (value < Eps)
            {
                value = Eps;
            }
            jScale[i] = Math.Sqrt(value);
        }

        var fNew = Vector<double>.Build.Dense(f.Count);
        for (int i = 0; i < f.Count; i++)
        {
            fNew[i] = f[i] * rho1[i] / jScale[i];
        }

        return (LeftMultiply(J, jScale), fNew);
    }

    // ------------------------------------------------------------------
    // Verbosity
    // ------------------------------------------------------------------

    /// <summary>Port of <c>print_header_nonlinear</c>.</summary>
    public static void PrintHeaderNonlinear()
    {
        Console.WriteLine(string.Format(
            "{0,15}{1,15}{2,15}{3,15}{4,15}{5,15}",
            "Iteration", "Total nfev", "Cost", "Cost reduction", "Step norm", "Optimality"));
    }

    /// <summary>Port of <c>print_iteration_nonlinear</c>.</summary>
    public static void PrintIterationNonlinear(
        int iteration, int nfev, double cost, double? costReduction,
        double? stepNorm, double optimality)
    {
        string reduction = costReduction is null
            ? new string(' ', 15)
            : $"{costReduction.Value,15:E2}";
        string step = stepNorm is null
            ? new string(' ', 15)
            : $"{stepNorm.Value,15:E2}";

        Console.WriteLine($"{iteration,15}{nfev,15}{cost,15:E4}{reduction}{step}{optimality,15:E2}");
    }

    /// <summary>Infinity norm (max absolute component) of a vector.</summary>
    public static double NormInf(Vector<double> x) => x.InfinityNorm();

    /// <summary>All components finite?</summary>
    public static bool AllFinite(Vector<double> x)
    {
        for (int i = 0; i < x.Count; i++)
        {
            if (!double.IsFinite(x[i]))
            {
                return false;
            }
        }
        return true;
    }
}
