using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp;

/// <summary>
/// Port of the finite-difference machinery SciPy uses when <c>jac</c> is not
/// callable (<c>scipy.optimize._numdiff</c>): relative steps, sign handling,
/// and the bound-aware switch between forward and central schemes.
/// </summary>
internal static class FiniteDifferences
{
    private const double Eps = 2.220446049250313e-16;

    private static double EpsForMethod(JacobianMethod method) => method switch
    {
        JacobianMethod.TwoPoint => Math.Sqrt(Eps),
        JacobianMethod.ThreePoint => Math.Pow(Eps, 1.0 / 3.0),
        _ => Math.Sqrt(Eps),
    };

    /// <summary>Estimates J(x) with shape (m, n) by finite differences.</summary>
    public static Matrix<double> Jacobian(
        Func<Vector<double>, Vector<double>> fun,
        Vector<double> x,
        Vector<double> f0,
        JacobianMethod method,
        double? relStep,
        Vector<double> lb,
        Vector<double> ub)
    {
        int n = x.Count;
        int m = f0.Count;
        double rstep = EpsForMethod(method);

        // sign_x0 = (x0 >= 0) * 2 - 1, i.e. +1 also when x0 == 0.
        var h = new double[n];
        for (int i = 0; i < n; i++)
        {
            double signX = x[i] >= 0 ? 1.0 : -1.0;
            double defaultStep = rstep * signX * Math.Max(1.0, Math.Abs(x[i]));
            if (relStep is null)
            {
                h[i] = defaultStep;
            }
            else
            {
                double absStep = relStep.Value * signX * Math.Abs(x[i]);
                double dx = (x[i] + absStep) - x[i];
                h[i] = dx == 0 ? defaultStep : absStep;
            }
        }

        bool[] useOneSided;
        if (method == JacobianMethod.TwoPoint)
        {
            AdjustOneSided(x, h, 1.0, lb, ub);
            useOneSided = new bool[n];
            Array.Fill(useOneSided, true);
        }
        else
        {
            useOneSided = AdjustTwoSided(x, h, 1.0, lb, ub);
        }

        var J = Matrix<double>.Build.Dense(m, n);

        for (int j = 0; j < n; j++)
        {
            var x1 = x.Clone();
            var x2 = x.Clone();
            Vector<double> f1;
            Vector<double> f2;
            double dx;

            if (method == JacobianMethod.TwoPoint)
            {
                x1[j] = x[j] + h[j];
                f1 = fun(x1);
                dx = x1[j] - x[j];
                for (int i = 0; i < m; i++)
                {
                    J[i, j] = (f1[i] - f0[i]) / dx;
                }
                continue;
            }

            if (useOneSided[j])
            {
                x1[j] = x[j] + h[j];
                x2[j] = x[j] + 2 * h[j];
                f1 = fun(x1);
                f2 = fun(x2);
                dx = x2[j] - x[j];
                for (int i = 0; i < m; i++)
                {
                    J[i, j] = (-3.0 * f0[i] + 4 * f1[i] - f2[i]) / dx;
                }
            }
            else
            {
                x1[j] = x[j] - h[j];
                x2[j] = x[j] + h[j];
                f1 = fun(x1);
                f2 = fun(x2);
                dx = x2[j] - x1[j];
                for (int i = 0; i < m; i++)
                {
                    J[i, j] = (f2[i] - f1[i]) / dx;
                }
            }
        }

        return J;
    }

    /// <summary>
    /// Port of <c>_adjust_scheme_to_bounds</c> for <c>scheme='1-sided'</c>
    /// (used by the 2-point scheme).
    /// </summary>
    private static void AdjustOneSided(
        Vector<double> x0, double[] h, double numSteps, Vector<double> lb, Vector<double> ub)
    {
        if (AllInfinite(lb, ub))
        {
            return;
        }

        var hTotal = new double[h.Length];
        for (int i = 0; i < h.Length; i++)
        {
            hTotal[i] = h[i] * numSteps;
        }

        for (int i = 0; i < h.Length; i++)
        {
            double lowerDist = x0[i] - lb[i];
            double upperDist = ub[i] - x0[i];

            double x = x0[i] + hTotal[i];
            bool violated = x < lb[i] || x > ub[i];
            bool fitting = Math.Abs(hTotal[i]) <= Math.Max(lowerDist, upperDist);

            if (violated && fitting)
            {
                h[i] *= -1;
            }
            else if (!fitting)
            {
                if (upperDist >= lowerDist)
                {
                    h[i] = upperDist / numSteps;
                }
                else
                {
                    h[i] = -lowerDist / numSteps;
                }
            }
        }
    }

    /// <summary>
    /// Port of <c>_adjust_scheme_to_bounds</c> for <c>scheme='2-sided'</c>
    /// (used by the 3-point scheme).  Returns the per-variable one-sided flag.
    /// </summary>
    private static bool[] AdjustTwoSided(
        Vector<double> x0, double[] h, double numSteps, Vector<double> lb, Vector<double> ub)
    {
        var useOneSided = new bool[h.Length];
        for (int i = 0; i < h.Length; i++)
        {
            h[i] = Math.Abs(h[i]);
        }

        if (AllInfinite(lb, ub))
        {
            return useOneSided;
        }

        var hTotal = new double[h.Length];
        for (int i = 0; i < h.Length; i++)
        {
            hTotal[i] = h[i] * numSteps;
        }

        var hAdjusted = (double[])h.Clone();

        for (int i = 0; i < h.Length; i++)
        {
            double lowerDist = x0[i] - lb[i];
            double upperDist = ub[i] - x0[i];

            bool central = lowerDist >= hTotal[i] && upperDist >= hTotal[i];
            if (central)
            {
                continue;
            }

            if (upperDist >= lowerDist)
            {
                hAdjusted[i] = Math.Min(h[i], 0.5 * upperDist / numSteps);
                useOneSided[i] = true;
            }
            else
            {
                hAdjusted[i] = -Math.Min(h[i], 0.5 * lowerDist / numSteps);
                useOneSided[i] = true;
            }
        }

        for (int i = 0; i < h.Length; i++)
        {
            double lowerDist = x0[i] - lb[i];
            double upperDist = ub[i] - x0[i];
            bool central = lowerDist >= hTotal[i] && upperDist >= hTotal[i];
            double minDist = Math.Min(upperDist, lowerDist) / numSteps;
            if (!central && Math.Abs(hAdjusted[i]) <= minDist)
            {
                hAdjusted[i] = minDist;
                useOneSided[i] = false;
            }
        }

        Array.Copy(hAdjusted, h, h.Length);
        return useOneSided;
    }

    private static bool AllInfinite(Vector<double> lb, Vector<double> ub)
    {
        for (int i = 0; i < lb.Count; i++)
        {
            if (lb[i] != double.NegativeInfinity || ub[i] != double.PositiveInfinity)
            {
                return false;
            }
        }
        return true;
    }
}
