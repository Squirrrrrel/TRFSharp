using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp.Core;

namespace TRF.Sharp;

/// <summary>
/// Bounded nonlinear least-squares minimization — a faithful C# port of
/// <c>scipy.optimize.least_squares(..., method='trf')</c>.
/// </summary>
/// <remarks>
/// Minimizes 0.5·Σ ρ(fᵢ(x)²) subject to <c>lb &lt;= x &lt;= ub</c> using the
/// Trust Region Reflective algorithm.  With <see cref="CurveFit"/> this is the
/// direct equivalent of <c>scipy.optimize.curve_fit</c> with bounds.
/// </remarks>
public static class LeastSquares
{
    /// <summary>
    /// Minimizes the residual vector returned by <paramref name="fun"/>.
    /// </summary>
    /// <param name="fun">Residual function: maps x (length n) to residuals (length m).</param>
    /// <param name="x0">Initial guess; for 'trf' it is slightly adjusted to lie strictly inside the bounds.</param>
    /// <param name="options">Solver options (bounds, Jacobian, tolerances, loss, …).</param>
    public static LeastSquaresResult Solve(
        Func<Vector<double>, Vector<double>> fun,
        Vector<double> x0,
        LeastSquaresOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        ArgumentNullException.ThrowIfNull(x0);

        options ??= new LeastSquaresOptions();

        int n = x0.Count;
        if (n == 0)
        {
            throw new ArgumentException("`x0` must not be empty.", nameof(x0));
        }

        var lb = BuildBounds(options.LowerBounds, n, double.NegativeInfinity, nameof(options.LowerBounds), n);
        var ub = BuildBounds(options.UpperBounds, n, double.PositiveInfinity, nameof(options.UpperBounds), n);

        for (int i = 0; i < n; i++)
        {
            if (lb[i] >= ub[i])
            {
                throw new ArgumentException(
                    "Each lower bound must be strictly less than each upper bound.");
            }
        }

        if (!Common.InBounds(x0, lb, ub))
        {
            throw new ArgumentException("Initial guess is outside of provided bounds");
        }

        XScaleSpec xScale;
        if (options.XScaleJac)
        {
            xScale = XScaleSpec.Jac();
        }
        else if (options.XScale is null)
        {
            xScale = XScaleSpec.Ones(n);
        }
        else
        {
            if (options.XScale.Length != n)
            {
                throw new ArgumentException("Inconsistent shapes between `x_scale` and `x0`.");
            }
            var values = Vector<double>.Build.DenseOfArray(options.XScale);
            for (int i = 0; i < n; i++)
            {
                if (!double.IsFinite(values[i]) || values[i] <= 0)
                {
                    throw new ArgumentException("`x_scale` must be 'jac' or array_like with positive numbers.");
                }
            }
            xScale = new XScaleSpec { IsJac = false, Values = values };
        }

        double ftol = options.Ftol ?? 0.0;
        double xtol = options.Xtol ?? 0.0;
        double gtol = options.Gtol ?? 0.0;
        if (ftol < Common.Eps && xtol < Common.Eps && gtol < Common.Eps)
        {
            throw new ArgumentException(
                "At least one of the tolerances must be higher than machine epsilon.");
        }

        if (options.MaxNfev is <= 0)
        {
            throw new ArgumentException("`max_nfev` must be None or positive integer.");
        }

        var cache = new FunCache(fun);
        var x0Feasible = Common.MakeStrictlyFeasible(x0, lb, ub);
        var f0 = cache.Evaluate(x0Feasible);

        if (!Common.AllFinite(f0))
        {
            throw new ArgumentException("Residuals are not finite in the initial point.");
        }

        int m = f0.Count;

        Loss? loss = options.Loss is LinearLoss ? null : options.Loss;
        double initialCost = loss is null
            ? 0.5 * f0.DotProduct(f0)
            : LossRuntime.CostOnly(loss, options.FScale, f0);

        var trSolver = options.TrustRegionSolver == TrustRegionSolver.Auto
            ? TrustRegionSolver.Exact
            : options.TrustRegionSolver;

        Func<Vector<double>, Matrix<double>> jac;
        if (options.Jacobian is not null)
        {
            var analytic = options.Jacobian;
            jac = analytic;
        }
        else
        {
            jac = x => FiniteDifferences.Jacobian(
                fun, x, cache.CurrentResiduals(x), options.JacobianMethod,
                options.DiffStep, lb, ub);
        }

        var j0 = jac(x0Feasible);
        if (j0.RowCount != m || j0.ColumnCount != n)
        {
            throw new ArgumentException(
                $"The return value of `jac` has wrong shape: expected ({m}, {n}), actual ({j0.RowCount}, {j0.ColumnCount}).");
        }

        var result = Trf.Solve(
            cache.Evaluate,
            jac,
            x0Feasible,
            f0,
            j0,
            lb,
            ub,
            ftol,
            xtol,
            gtol,
            options.MaxNfev,
            xScale,
            loss,
            options.FScale,
            trSolver,
            options.ToTrfOptions(),
            options.Verbose,
            options.Callback);

        if (options.Verbose >= 1)
        {
            Console.WriteLine(result.Message);
            Console.WriteLine(
                $"Function evaluations {result.Nfev}, initial cost {initialCost:E4}, " +
                $"final cost {result.Cost:E4}, first-order optimality {result.Optimality:E2}.");
        }

        return result;
    }

    /// <summary>Convenience overload taking plain arrays.</summary>
    public static LeastSquaresResult Solve(
        Func<double[], double[]> fun,
        double[] x0,
        LeastSquaresOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(fun);
        return Solve(
            x => Vector<double>.Build.DenseOfArray(fun(x.ToArray())),
            Vector<double>.Build.DenseOfArray(x0),
            options);
    }

    private static Vector<double> BuildBounds(
        double[]? bound, int n, double fill, string name, int expected)
    {
        if (bound is null)
        {
            return Vector<double>.Build.Dense(n, fill);
        }

        if (bound.Length == 1)
        {
            return Vector<double>.Build.Dense(n, bound[0]);
        }

        if (bound.Length != expected)
        {
            throw new ArgumentException($"Inconsistent shapes between `{name}` and `x0`.");
        }

        return Vector<double>.Build.DenseOfArray(bound);
    }

    /// <summary>
    /// Remembers the most recent residual vector so that a finite-difference
    /// Jacobian can reuse it exactly as SciPy's <c>VectorFunction</c> does.
    /// </summary>
    private sealed class FunCache
    {
        private readonly Func<Vector<double>, Vector<double>> _fun;
        private Vector<double>? _x;
        private Vector<double>? _f;

        public FunCache(Func<Vector<double>, Vector<double>> fun) => _fun = fun;

        public Vector<double> Evaluate(Vector<double> x)
        {
            var f = _fun(x);
            _x = x.Clone();
            _f = f;
            return f;
        }

        public Vector<double> CurrentResiduals(Vector<double> x)
        {
            if (_x is not null && _f is not null && Same(_x, x))
            {
                return _f;
            }
            return _fun(x);
        }

        private static bool Same(Vector<double> a, Vector<double> b)
        {
            if (a.Count != b.Count)
            {
                return false;
            }
            for (int i = 0; i < a.Count; i++)
            {
                if (!a[i].Equals(b[i]))
                {
                    return false;
                }
            }
            return true;
        }
    }
}
