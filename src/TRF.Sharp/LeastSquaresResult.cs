using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp;

/// <summary>
/// Outcome of a nonlinear least-squares minimization.  Mirrors the fields of
/// <c>scipy.optimize.OptimizeResult</c> as returned by
/// <c>scipy.optimize.least_squares</c>.
/// </summary>
public sealed class LeastSquaresResult
{
    /// <summary>Solution found.</summary>
    public Vector<double> X { get; set; } = Vector<double>.Build.Dense(0);

    /// <summary>Value of the cost function (0.5·Σ fᵢ²) at the solution.</summary>
    public double Cost { get; set; }

    /// <summary>Vector of residuals at the solution.</summary>
    public Vector<double> Fun { get; set; } = Vector<double>.Build.Dense(0);

    /// <summary>
    /// Modified Jacobian matrix at the solution, in the sense that Jᵀ·J is a
    /// Gauss–Newton approximation of the Hessian of the cost function.
    /// </summary>
    public Matrix<double>? Jac { get; set; }

    /// <summary>Gradient of the cost function at the solution.</summary>
    public Vector<double> Grad { get; set; } = Vector<double>.Build.Dense(0);

    /// <summary>
    /// First-order optimality measure.  In constrained problems this is the
    /// quantity compared against <c>gtol</c> during iterations.
    /// </summary>
    public double Optimality { get; set; }

    /// <summary>
    /// Per-component activity indicator: 0 = inactive, −1 = lower bound active,
    /// +1 = upper bound active.
    /// </summary>
    public int[] ActiveMask { get; set; } = Array.Empty<int>();

    /// <summary>Number of function evaluations (excluding Jacobian differencing).</summary>
    public int Nfev { get; set; }

    /// <summary>Number of Jacobian evaluations.</summary>
    public int Njev { get; set; }

    /// <summary>Number of iterations.</summary>
    public int Nit { get; set; }

    /// <summary>Termination reason code (see <see cref="Message"/>).</summary>
    public int Status { get; set; }

    /// <summary>Verbal description of the termination reason.</summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>True when one of the convergence criteria is satisfied (status &gt; 0).</summary>
    public bool Success { get; set; }

    /// <summary>Convenience: the solution as a plain array.</summary>
    public double[] Parameters => X.ToArray();
}
