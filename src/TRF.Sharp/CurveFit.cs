using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp;

/// <summary>Result of a bounded curve fit.</summary>
public sealed class CurveFitResult
{
    /// <summary>Optimal parameters.</summary>
    public Vector<double> Popt { get; init; } = Vector<double>.Build.Dense(0);

    /// <summary>Estimated covariance matrix of <see cref="Popt"/>.</summary>
    public Matrix<double> Pcov { get; init; } = Matrix<double>.Build.Dense(0, 0);

    /// <summary>One standard-deviation errors, i.e. sqrt(diag(Pcov)).</summary>
    public Vector<double> Perp { get; init; } = Vector<double>.Build.Dense(0);

    /// <summary>True when the covariance could not be estimated (m ≤ n).</summary>
    public bool CovarianceUnreliable { get; init; }

    /// <summary>True when at least one parameter sits on a bound (covariance is then only indicative).</summary>
    public bool HasActiveBounds { get; init; }

    /// <summary>Underlying least-squares result.</summary>
    public LeastSquaresResult Result { get; init; } = new();

    /// <summary>Optimal parameters as a plain array.</summary>
    public double[] Parameters => Popt.ToArray();
}

/// <summary>
/// Bounded curve fitting — the C# counterpart of
/// <c>scipy.optimize.curve_fit(..., bounds=...)</c>, which uses TRF internally.
/// </summary>
public static class CurveFit
{
    /// <summary>
    /// Fits <paramref name="model"/> to <paramref name="ydata"/> by minimizing
    /// Σ ((model(xᵢ, p) − yᵢ) / σᵢ)² over p, subject to the parameter bounds.
    /// </summary>
    /// <param name="model">Model function: (x, parameters) → y.</param>
    /// <param name="xdata">Predictor values.</param>
    /// <param name="ydata">Observations, same length as <paramref name="xdata"/>.</param>
    /// <param name="p0">Initial guess for the parameters.</param>
    /// <param name="lowerBounds">Per-parameter lower bounds (null = −∞).</param>
    /// <param name="upperBounds">Per-parameter upper bounds (null = +∞).</param>
    /// <param name="sigma">Uncertainties of <paramref name="ydata"/> (null = 1).</param>
    /// <param name="absoluteSigma">True if <paramref name="sigma"/> is absolute (then pcov is not rescaled).</param>
    /// <param name="options">Optional solver options (the bounds are taken from the dedicated parameters).</param>
    public static CurveFitResult Fit(
        Func<double, Vector<double>, double> model,
        Vector<double> xdata,
        Vector<double> ydata,
        Vector<double> p0,
        double[]? lowerBounds = null,
        double[]? upperBounds = null,
        double[]? sigma = null,
        bool absoluteSigma = false,
        LeastSquaresOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (xdata.Count != ydata.Count)
        {
            throw new ArgumentException("`xdata` and `ydata` must have the same length.");
        }

        var settings = Clone(options);
        if (lowerBounds is not null)
        {
            settings.LowerBounds = lowerBounds;
        }
        if (upperBounds is not null)
        {
            settings.UpperBounds = upperBounds;
        }

        int m = xdata.Count;
        int n = p0.Count;

        Vector<double> Residuals(Vector<double> p)
        {
            var r = Vector<double>.Build.Dense(m);
            for (int i = 0; i < m; i++)
            {
                double value = model(xdata[i], p) - ydata[i];
                if (sigma is not null)
                {
                    value /= sigma[i];
                }
                r[i] = value;
            }
            return r;
        }

        var result = LeastSquares.Solve(Residuals, p0, settings);
        var (pcov, unreliable) = Covariance(result, m, n, absoluteSigma);

        bool activeBounds = false;
        foreach (int mask in result.ActiveMask)
        {
            if (mask != 0)
            {
                activeBounds = true;
                break;
            }
        }

        var perr = Vector<double>.Build.Dense(n, i => Math.Sqrt(pcov[i, i]));

        return new CurveFitResult
        {
            Popt = result.X,
            Pcov = pcov,
            Perp = perr,
            CovarianceUnreliable = unreliable,
            HasActiveBounds = activeBounds,
            Result = result,
        };
    }

    /// <summary>Convenience overload with plain arrays and a <c>double[]</c> parameter vector.</summary>
    public static CurveFitResult Fit(
        Func<double, double[], double> model,
        double[] xdata,
        double[] ydata,
        double[] p0,
        double[]? lowerBounds = null,
        double[]? upperBounds = null,
        double[]? sigma = null,
        bool absoluteSigma = false,
        LeastSquaresOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Fit(
            (x, p) => model(x, p.ToArray()),
            Vector<double>.Build.DenseOfArray(xdata),
            Vector<double>.Build.DenseOfArray(ydata),
            Vector<double>.Build.DenseOfArray(p0),
            lowerBounds,
            upperBounds,
            sigma,
            absoluteSigma,
            options);
    }

    /// <summary>
    /// Port of the covariance computation in SciPy's <c>curve_fit</c>: a
    /// Moore–Penrose pseudo-inverse of JᵀJ via SVD, rescaled by the residual
    /// variance unless <c>absolute_sigma</c> is set.
    /// </summary>
    private static (Matrix<double> pcov, bool unreliable) Covariance(
        LeastSquaresResult result, int m, int n, bool absoluteSigma)
    {
        var J = result.Jac ?? Matrix<double>.Build.Dense(m, n);
        var svd = J.Svd();
        var s = svd.S;
        var VT = svd.VT;

        double threshold = 2.220446049250313e-16 * Math.Max(J.RowCount, J.ColumnCount) * s[0];
        var kept = new List<int>();
        for (int i = 0; i < s.Count; i++)
        {
            if (s[i] > threshold)
            {
                kept.Add(i);
            }
        }

        var pcov = Matrix<double>.Build.Dense(n, n);
        foreach (int i in kept)
        {
            var vi = VT.Row(i);
            double weight = 1.0 / (s[i] * s[i]);
            for (int a = 0; a < n; a++)
            {
                for (int b = 0; b < n; b++)
                {
                    pcov[a, b] += vi[a] * vi[b] * weight;
                }
            }
        }

        if (absoluteSigma)
        {
            return (pcov, false);
        }

        if (m > n)
        {
            double cost = result.Fun.DotProduct(result.Fun);
            double sSq = cost / (m - n);
            return (pcov * sSq, false);
        }

        // SciPy fills the covariance with inf and warns in this case.
        pcov = Matrix<double>.Build.Dense(n, n, double.PositiveInfinity);
        return (pcov, true);
    }

    private static LeastSquaresOptions Clone(LeastSquaresOptions? options)
    {
        if (options is null)
        {
            return new LeastSquaresOptions();
        }

        return new LeastSquaresOptions
        {
            LowerBounds = options.LowerBounds,
            UpperBounds = options.UpperBounds,
            Jacobian = options.Jacobian,
            JacobianMethod = options.JacobianMethod,
            DiffStep = options.DiffStep,
            Ftol = options.Ftol,
            Xtol = options.Xtol,
            Gtol = options.Gtol,
            XScale = options.XScale,
            XScaleJac = options.XScaleJac,
            Loss = options.Loss,
            FScale = options.FScale,
            TrustRegionSolver = options.TrustRegionSolver,
            MaxNfev = options.MaxNfev,
            Verbose = options.Verbose,
            LsmrDamp = options.LsmrDamp,
            LsmrAtol = options.LsmrAtol,
            LsmrBtol = options.LsmrBtol,
            LsmrConlim = options.LsmrConlim,
            LsmrMaxIter = options.LsmrMaxIter,
            LsmrRegularize = options.LsmrRegularize,
            Callback = options.Callback,
        };
    }
}
