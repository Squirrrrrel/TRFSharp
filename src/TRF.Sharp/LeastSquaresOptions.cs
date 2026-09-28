using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp;

/// <summary>How the Jacobian matrix is obtained.</summary>
public enum JacobianMethod
{
    /// <summary>Forward (2-point) finite differences, SciPy's default.</summary>
    TwoPoint = 0,

    /// <summary>Central (3-point) finite differences: more accurate, twice the cost.</summary>
    ThreePoint = 1,
}

/// <summary>Trust-region subproblem solver, equivalent to SciPy's <c>tr_solver</c>.</summary>
public enum TrustRegionSolver
{
    /// <summary>Chosen from the Jacobian type (default: <see cref="Exact"/>).</summary>
    Auto = 0,

    /// <summary>SVD-based accurate solver — the natural choice for dense problems.</summary>
    Exact = 1,

    /// <summary>2-D subspace method driven by LSMR — for large / sparse problems.</summary>
    Lsmr = 2,
}

/// <summary>
/// Options for <see cref="LeastSquares"/> and <see cref="CurveFit"/>.  Defaults
/// match <c>scipy.optimize.least_squares(method='trf')</c>.
/// </summary>
public sealed class LeastSquaresOptions
{
    /// <summary>Lower bounds (per variable). Null or <c>-Infinity</c> means unbounded.</summary>
    public double[]? LowerBounds { get; set; }

    /// <summary>Upper bounds (per variable). Null or <c>+Infinity</c> means unbounded.</summary>
    public double[]? UpperBounds { get; set; }

    /// <summary>Analytic Jacobian J(x) with shape (m, n). When null, finite differences are used.</summary>
    public Func<Vector<double>, Matrix<double>>? Jacobian { get; set; }

    /// <summary>Finite-difference scheme used when <see cref="Jacobian"/> is null.</summary>
    public JacobianMethod JacobianMethod { get; set; } = JacobianMethod.TwoPoint;

    /// <summary>
    /// Relative step for finite differences; the actual step is <c>x * DiffStep</c>.
    /// Null selects SciPy's "optimal" power of machine epsilon for the scheme.
    /// </summary>
    public double? DiffStep { get; set; }

    /// <summary>Tolerance on the change of the cost function (SciPy <c>ftol</c>).</summary>
    public double? Ftol { get; set; } = 1e-8;

    /// <summary>Tolerance on the change of the variables (SciPy <c>xtol</c>).</summary>
    public double? Xtol { get; set; } = 1e-8;

    /// <summary>Tolerance on the gradient / optimality measure (SciPy <c>gtol</c>).</summary>
    public double? Gtol { get; set; } = 1e-8;

    /// <summary>
    /// Characteristic scale of each variable (SciPy <c>x_scale</c> as an array).
    /// Mutually exclusive with <see cref="XScaleJac"/>.
    /// </summary>
    public double[]? XScale { get; set; }

    /// <summary>
    /// True selects <c>x_scale='jac'</c>: the scale is iteratively updated from the
    /// inverse norms of the Jacobian columns.
    /// </summary>
    public bool XScaleJac { get; set; }

    /// <summary>Robust loss function (SciPy <c>loss</c>). Defaults to <see cref="Loss.Linear"/>.</summary>
    public Loss Loss { get; set; } = Loss.Linear;

    /// <summary>Soft margin between inlier and outlier residuals (SciPy <c>f_scale</c>).</summary>
    public double FScale { get; set; } = 1.0;

    /// <summary>Trust-region subproblem solver.</summary>
    public TrustRegionSolver TrustRegionSolver { get; set; } = TrustRegionSolver.Auto;

    /// <summary>Maximum number of function evaluations; null means <c>100 * n</c>.</summary>
    public int? MaxNfev { get; set; }

    /// <summary>Verbosity: 0 silent, 1 termination report, 2 per-iteration progress.</summary>
    public int Verbose { get; set; }

    /// <summary>LSMR damping factor (only used with <see cref="TRF.Sharp.TrustRegionSolver.Lsmr"/>).</summary>
    public double LsmrDamp { get; set; }

    /// <summary>LSMR absolute tolerance on A.</summary>
    public double LsmrAtol { get; set; } = 1e-6;

    /// <summary>LSMR absolute tolerance on b.</summary>
    public double LsmrBtol { get; set; } = 1e-6;

    /// <summary>LSMR condition-number limit.</summary>
    public double LsmrConlim { get; set; } = 1e8;

    /// <summary>LSMR iteration limit; null means <c>min(m, n)</c>.</summary>
    public int? LsmrMaxIter { get; set; }

    /// <summary>
    /// Adds a regularization term to the normal equations when using LSMR
    /// (SciPy <c>tr_options={'regularize': True}</c>); improves convergence for
    /// rank-deficient Jacobians.
    /// </summary>
    public bool LsmrRegularize { get; set; } = true;

    /// <summary>
    /// Called after every iteration.  Returning true stops the optimizer
    /// (equivalent to SciPy's callback raising <c>StopIteration</c>, status −2).
    /// </summary>
    public Func<LeastSquaresResult, bool>? Callback { get; set; }

    internal TrfOptions ToTrfOptions() => new()
    {
        Damp = LsmrDamp,
        Atol = LsmrAtol,
        Btol = LsmrBtol,
        Conlim = LsmrConlim,
        MaxIter = LsmrMaxIter,
        Regularize = LsmrRegularize,
    };
}

/// <summary>Internal LSMR knobs passed down to the TRF driver.</summary>
internal sealed class TrfOptions
{
    public double Damp { get; init; }

    public double Atol { get; init; } = 1e-6;

    public double Btol { get; init; } = 1e-6;

    public double Conlim { get; init; } = 1e8;

    public int? MaxIter { get; init; }

    public bool Regularize { get; init; } = true;
}

/// <summary>Internal representation of the SciPy <c>x_scale</c> parameter.</summary>
internal sealed class XScaleSpec
{
    public bool IsJac { get; init; }

    public Vector<double> Values { get; init; } = Vector<double>.Build.Dense(0);

    public static XScaleSpec Ones(int n) =>
        new() { IsJac = false, Values = Vector<double>.Build.Dense(n, 1.0) };

    public static XScaleSpec Jac() => new() { IsJac = true };
}
