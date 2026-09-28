using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp;

/// <summary>
/// Robust loss function ρ(z), z = (fᵢ / f_scale)², exactly as defined in
/// <c>scipy.optimize._lsq.least_squares</c>.
/// </summary>
/// <remarks>
/// A loss must fill three arrays of length m: ρ itself (row 0), its first
/// derivative ρ′ (row 1) and its second derivative ρ″ (row 2).
/// </remarks>
public abstract class Loss
{
    /// <summary>ρ(z) = z — the standard least-squares problem.</summary>
    public static Loss Linear { get; } = new LinearLoss();

    /// <summary>ρ(z) = 2·(√(1+z) − 1) — smooth approximation of the ℓ1 loss.</summary>
    public static Loss SoftL1 { get; } = new SoftL1Loss();

    /// <summary>ρ(z) = z for z ≤ 1, else 2√z − 1.</summary>
    public static Loss Huber { get; } = new HuberLoss();

    /// <summary>ρ(z) = ln(1 + z) — strongly weakens outliers.</summary>
    public static Loss Cauchy { get; } = new CauchyLoss();

    /// <summary>ρ(z) = arctan(z) — bounds the loss of a single residual.</summary>
    public static Loss Arctan { get; } = new ArctanLoss();

    /// <summary>Builds a loss from a user supplied evaluator.</summary>
    public static Loss Custom(Func<Vector<double>, (double[] rho0, double[] rho1, double[] rho2)> evaluator) =>
        new CustomLoss(evaluator);

    /// <summary>Fills ρ, ρ′ and ρ″ for the vector <paramref name="z"/>.</summary>
    public abstract void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2);

    internal virtual bool IsLinear => false;
}

internal sealed class LinearLoss : Loss
{
    public override void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2)
    {
        for (int i = 0; i < z.Count; i++)
        {
            rho0[i] = z[i];
            rho1[i] = 1.0;
            rho2[i] = 0.0;
        }
    }

    internal override bool IsLinear => true;
}

internal sealed class SoftL1Loss : Loss
{
    public override void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2)
    {
        for (int i = 0; i < z.Count; i++)
        {
            double t = 1 + z[i];
            rho0[i] = 2 * (Math.Sqrt(t) - 1);
            rho1[i] = 1.0 / Math.Sqrt(t);
            rho2[i] = -0.5 * Math.Pow(t, -1.5);
        }
    }
}

internal sealed class HuberLoss : Loss
{
    public override void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2)
    {
        for (int i = 0; i < z.Count; i++)
        {
            if (z[i] <= 1)
            {
                rho0[i] = z[i];
                rho1[i] = 1.0;
                rho2[i] = 0.0;
            }
            else
            {
                rho0[i] = 2 * Math.Sqrt(z[i]) - 1;
                rho1[i] = 1.0 / Math.Sqrt(z[i]);
                rho2[i] = -0.5 * Math.Pow(z[i], -1.5);
            }
        }
    }
}

internal sealed class CauchyLoss : Loss
{
    public override void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2)
    {
        for (int i = 0; i < z.Count; i++)
        {
            double t = 1 + z[i];
            rho0[i] = Math.Log(1 + z[i]);
            rho1[i] = 1.0 / t;
            rho2[i] = -1.0 / (t * t);
        }
    }
}

internal sealed class ArctanLoss : Loss
{
    public override void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2)
    {
        for (int i = 0; i < z.Count; i++)
        {
            double t = 1 + z[i] * z[i];
            rho0[i] = Math.Atan(z[i]);
            rho1[i] = 1.0 / t;
            rho2[i] = -2 * z[i] / (t * t);
        }
    }
}

internal sealed class CustomLoss : Loss
{
    private readonly Func<Vector<double>, (double[] rho0, double[] rho1, double[] rho2)> _evaluator;

    public CustomLoss(Func<Vector<double>, (double[] rho0, double[] rho1, double[] rho2)> evaluator) =>
        _evaluator = evaluator ?? throw new ArgumentNullException(nameof(evaluator));

    public override void Evaluate(Vector<double> z, double[] rho0, double[] rho1, double[] rho2)
    {
        var result = _evaluator(z);
        Array.Copy(result.rho0, rho0, z.Count);
        Array.Copy(result.rho1, rho1, z.Count);
        Array.Copy(result.rho2, rho2, z.Count);
    }
}

/// <summary>Applies the SciPy <c>construct_loss_function</c> scaling.</summary>
internal static class LossRuntime
{
    public static Matrix<double> Rho(Loss loss, double fScale, Vector<double> f)
    {
        int m = f.Count;
        var z = Vector<double>.Build.Dense(m);
        for (int i = 0; i < m; i++)
        {
            double scaled = f[i] / fScale;
            z[i] = scaled * scaled;
        }

        var rho0 = new double[m];
        var rho1 = new double[m];
        var rho2 = new double[m];
        loss.Evaluate(z, rho0, rho1, rho2);

        double f2 = fScale * fScale;
        var rho = Matrix<double>.Build.Dense(3, m);
        for (int i = 0; i < m; i++)
        {
            rho[0, i] = rho0[i] * f2;
            rho[1, i] = rho1[i];
            rho[2, i] = rho2[i] / f2;
        }
        return rho;
    }

    public static double CostOnly(Loss loss, double fScale, Vector<double> f)
    {
        int m = f.Count;
        var z = Vector<double>.Build.Dense(m);
        for (int i = 0; i < m; i++)
        {
            double scaled = f[i] / fScale;
            z[i] = scaled * scaled;
        }

        var rho0 = new double[m];
        var rho1 = new double[m];
        var rho2 = new double[m];
        loss.Evaluate(z, rho0, rho1, rho2);

        double sum = 0.0;
        for (int i = 0; i < m; i++)
        {
            sum += rho0[i];
        }
        return 0.5 * fScale * fScale * sum;
    }
}
