// Nine-parameter nonlinear curve fitting with TRF.Sharp — the four ways you
// are likely to call it, on one synthetic three-peak spectrum.
//
//   model:  A1·G(t;c1,w1) + A2·G(t;c2,w2) + A3·G(t;c3,w3),  G = exp(-0.5 z²)
//   params: p = [A1, c1, w1, A2, c2, w2, A3, c3, w3]        (n = 9, m = 256)
//
// Run:  dotnet run -c Release --project examples/NineParamFit

using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp;

namespace NineParamFit;

internal static class Program
{
    private const int M = 256;
    private const int N = 9;

    private static readonly double[] Truth =
        { 1.00, 1.50, 0.55, 0.65, 4.00, 0.90, 0.40, 6.80, 0.45 };

    // A deliberately mediocre starting point — peaks guessed a bit off,
    // widths guessed too wide. TRF converges anyway.
    private static readonly double[] P0 =
        { 1.20, 1.35, 0.75, 0.50, 4.25, 1.10, 0.30, 6.55, 0.60 };

    private static readonly double[] Lower =
        { 0.0, 0.0, 0.05, 0.0, 2.0, 0.05, 0.0, 5.0, 0.05 };

    private static readonly double[] Upper =
        { 5.0, 3.0, 2.00, 5.0, 6.0, 2.00, 5.0, 9.0, 2.00 };

    private static readonly double[] T = new double[M];
    private static readonly double[] Y = new double[M];

    private static void Main()
    {
        MakeData();

        Console.WriteLine("nine-parameter three-peak fit, "
                          + $"m = {M} data points, n = {N} parameters");
        Console.WriteLine();

        Section1FiniteDifferences();
        Section2AnalyticJacobian();
        Section3Outliers();
        Section4RawLeastSquares();

        Console.WriteLine();
        Console.WriteLine("Truth was  "
                          + string.Join("  ", Truth.Select(v => v.ToString("F4"))));
    }

    // ---------------------------------------------------------------- 1
    private static void Section1FiniteDifferences()
    {
        Console.WriteLine("[1] CurveFit.Fit, finite-difference Jacobian");

        var fit = CurveFit.Fit(Model, T, Y, P0, Lower, Upper);

        Console.WriteLine($"    popt   {Row(fit.Parameters)}");
        Console.WriteLine($"    perr   {Row(fit.Perp)}");
        Console.WriteLine($"    cost {fit.Result.Cost:E3}  nfev {fit.Result.Nfev}"
                          + $"  status {fit.Result.Status}  {fit.Result.Message}");
        Console.WriteLine($"    time   {Time(() => CurveFit.Fit(Model, T, Y, P0, Lower, Upper)):F2} ms");
        Console.WriteLine();
    }

    // ---------------------------------------------------------------- 2
    private static void Section2AnalyticJacobian()
    {
        Console.WriteLine("[2] CurveFit.Fit, analytic Jacobian (recommended)");

        var options = new LeastSquaresOptions { Jacobian = Jacobian };

        var tv = Vector<double>.Build.DenseOfArray(T);
        var yv = Vector<double>.Build.DenseOfArray(Y);
        var p0v = Vector<double>.Build.DenseOfArray(P0);

        var fit = CurveFit.Fit(ModelVec, tv, yv, p0v, Lower, Upper, options: options);

        // same problem without the covariance step, to price that step
        Vector<double> Residual(Vector<double> p) =>
            Vector<double>.Build.Dense(M, i => ModelVec(T[i], p) - Y[i]);
        var bare = new LeastSquaresOptions
        {
            Jacobian = Jacobian, LowerBounds = Lower, UpperBounds = Upper,
        };

        Console.WriteLine($"    popt   {Row(fit.Parameters)}");
        Console.WriteLine($"    perr   {Row(fit.Perp)}");
        Console.WriteLine($"    cost {fit.Result.Cost:E3}  nfev {fit.Result.Nfev}"
                          + $"  status {fit.Result.Status}  {fit.Result.Message}");
        Console.WriteLine($"    time   {Time(() => CurveFit.Fit(ModelVec, tv, yv, p0v, Lower, Upper, options: options)):F2} ms"
                          + $"  (solve only: "
                          + $"{Time(() => LeastSquares.Solve(Residual, p0v, bare)):F2} ms)");
        Console.WriteLine();
    }

    // ---------------------------------------------------------------- 3
    private static void Section3Outliers()
    {
        Console.WriteLine("[3] Same data with 8% outliers — robust loss");

        var rng = new Random(7);
        var yOut = (double[])Y.Clone();
        for (int i = 0; i < M; i++)
        {
            if (rng.NextDouble() < 0.08)
            {
                yOut[i] += 0.35 * (rng.NextDouble() - 0.5);
            }
        }

        foreach ((string name, Loss loss) in new[]
                 { ("linear", Loss.Linear), ("soft_l1", Loss.SoftL1),
                   ("huber", Loss.Huber), ("cauchy", Loss.Cauchy) })
        {
            var options = new LeastSquaresOptions { Loss = loss, FScale = 0.02 };
            var fit = CurveFit.Fit(Model, T, yOut, P0, Lower, Upper,
                                   options: options);
            Console.WriteLine($"    {name,-8} cost {fit.Result.Cost:E3}"
                              + $"  A2 {fit.Parameters[3]:F4}"
                              + $"  c2 {fit.Parameters[4]:F4}"
                              + $"  nfev {fit.Result.Nfev}");
        }

        Console.WriteLine();
    }

    // ---------------------------------------------------------------- 4
    private static void Section4RawLeastSquares()
    {
        Console.WriteLine("[4] LeastSquares.Solve directly (custom residual)");

        var y = Vector<double>.Build.DenseOfArray(Y);
        var t = Vector<double>.Build.DenseOfArray(T);

        // weight the wings of the spectrum down
        Vector<double> Residual(Vector<double> p)
        {
            var r = Vector<double>.Build.Dense(M);
            for (int i = 0; i < M; i++)
            {
                r[i] = (Model(t[i], p) - y[i]) * Weight(i);
            }

            return r;
        }

        var options = new LeastSquaresOptions
        {
            LowerBounds = Lower,
            UpperBounds = Upper,
            XScale = new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1 },
            Verbose = 0,
        };

        var r = LeastSquares.Solve(Residual, Vector<double>.Build.DenseOfArray(P0),
                                   options);
        Console.WriteLine($"    x      {Row(r.Parameters)}");
        Console.WriteLine($"    cost {r.Cost:E3}  optimality {r.Optimality:E2}"
                          + $"  nit {r.Nit}  status {r.Status}");
        Console.WriteLine();
    }

    // -------------------------------------------------------------- model
    private static double Model(double t, double[] p)
    {
        double z1 = (t - p[1]) / p[2];
        double z2 = (t - p[4]) / p[5];
        double z3 = (t - p[7]) / p[8];
        return p[0] * Math.Exp(-0.5 * z1 * z1)
             + p[3] * Math.Exp(-0.5 * z2 * z2)
             + p[6] * Math.Exp(-0.5 * z3 * z3);
    }

    private static double Model(double t, Vector<double> p) => Model(t, p.ToArray());

    private static double ModelVec(double t, Vector<double> p)
    {
        double z1 = (t - p[1]) / p[2];
        double z2 = (t - p[4]) / p[5];
        double z3 = (t - p[7]) / p[8];
        return p[0] * Math.Exp(-0.5 * z1 * z1)
             + p[3] * Math.Exp(-0.5 * z2 * z2)
             + p[6] * Math.Exp(-0.5 * z3 * z3);
    }

    /// <summary>d(model)/dp as an m×9 matrix — the residual Jacobian when
    /// σ = 1. Rows are data points, columns are parameters.</summary>
    private static Matrix<double> Jacobian(Vector<double> p)
    {
        var J = Matrix<double>.Build.Dense(M, N);
        for (int i = 0; i < M; i++)
        {
            double t = T[i];
            for (int k = 0; k < 3; k++)
            {
                double z = (t - p[3 * k + 1]) / p[3 * k + 2];
                double g = Math.Exp(-0.5 * z * z);
                J[i, 3 * k] = g;                          // d/dA
                J[i, 3 * k + 1] = p[3 * k] * g * z / p[3 * k + 2];   // d/dc
                J[i, 3 * k + 2] = p[3 * k] * g * z * z / p[3 * k + 2]; // d/dw
            }
        }

        return J;
    }

    private static double Weight(int i)
    {
        double t = T[i];
        return t < 2.0 || t > 8.0 ? 0.5 : 1.0;
    }

    // --------------------------------------------------------------- utils
    private static void MakeData()
    {
        var rng = new Random(4242);
        for (int i = 0; i < M; i++)
        {
            T[i] = -1.0 + 10.0 * i / (M - 1);
            Y[i] = Model(T[i], Truth) + 0.004 * NextNormal(rng);
        }
    }

    private static string Row(IEnumerable<double> v) =>
        string.Join("  ", v.Select(x => x.ToString("F4")));

    private static double Time<T>(Func<T> fn)
    {
        for (int i = 0; i < 3; i++)
        {
            _ = fn();           // warm-up: JIT + tiered PGO
        }

        var best = double.MaxValue;
        for (int i = 0; i < 9; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = fn();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }

        return best;
    }

    private static double NextNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
