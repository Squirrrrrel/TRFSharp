// MathNet side of the nine-parameter micro-benchmark.
//
// Mirrors bench/micro_bench.py: one residual evaluation, one economy SVD of
// the augmented Jacobian, one 2-point finite-difference Jacobian.  Same m
// list, same iteration counts, best-of-N timing.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp.Core;

namespace TRF.Sharp.Bench.Micro9;

internal static class Program
{
    private const int N = 9;
    private const int Iters = 400;

    private static readonly int[] Ms = { 32, 64, 128, 256, 512, 1024 };

    private static readonly double[] P =
        { 1.0, 1.5, 0.55, 0.65, 4.0, 0.90, 0.40, 6.8, 0.45 };

    private static int Main(string[] args)
    {
        string outPath = args.Length > 0 ? args[0] : "results_micro_csharp.json";
        var rows = new List<Dictionary<string, object>>();

        foreach (int m in Ms)
        {
            var rng = new Random(1000 + m);
            var t = Vector<double>.Build.Dense(m);
            var y = Vector<double>.Build.Dense(m);
            for (int i = 0; i < m; i++)
            {
                t[i] = -1.0 + 10.0 * i / (m - 1);
                y[i] = Model(t[i], P) + 0.004 * NextNormal(rng);
            }

            // warm-up
            for (int i = 0; i < 20; i++)
            {
                _ = Residual(t, y, P);
                _ = FdJacobian(t, y, P, m);
                _ = Augmented(FdJacobian(t, y, P, m), m).Svd();
            }

            var J = FdJacobian(t, y, P, m);
            var aug = Augmented(J, m);
            var d = Vector<double>.Build.Dense(N, i => 1.0 + 0.1 * i);

            double rMin = Best(() => { _ = Residual(t, y, P); }, Iters);
            double sMin = Best(() => { _ = aug.Svd(); }, Iters);
            // what the library actually pays for a factorisation today
            double sLibMin = Best(() => { _ = ThinQr.FactorR(aug).Svd(); }, Iters);
            double fMin = Best(() => { _ = FdJacobian(t, y, P, m); },
                               Math.Max(20, Iters / 5));
            // the remaining O(m*n) work of one TRF iteration
            double augMin = Best(() => { _ = Augmented(J, m); }, Iters);
            double scaleMin = Best(() => { _ = ScaleColumns(J, d); }, Iters);
            double mvMin = Best(() => { _ = J.Multiply(d); }, Iters);

            rows.Add(new Dictionary<string, object>
            {
                ["m"] = m,
                ["n"] = N,
                ["residual_min"] = rMin,
                ["svd_min"] = sMin,
                ["svd_lib_min"] = sLibMin,
                ["fdjac_min"] = fMin,
                ["augment_min"] = augMin,
                ["scalecols_min"] = scaleMin,
                ["matvec_min"] = mvMin,
            });
            Console.WriteLine($"  m={m,-5} residual {rMin * 1e6,8:F2} us   "
                              + $"svd({m + N}x{N}) {sMin * 1e6,9:F2} us   "
                              + $"lib-qr {sLibMin * 1e6,8:F2} us   "
                              + $"fd-jac {fMin * 1e6,9:F2} us   "
                              + $"augment {augMin * 1e6,7:F2} us   "
                              + $"scale {scaleMin * 1e6,7:F2} us   "
                              + $"matvec {mvMin * 1e6,7:F2} us");
        }

        File.WriteAllText(outPath, JsonSerializer.Serialize(
            new Dictionary<string, object>
            {
                ["rows"] = rows,
                ["meta"] = new Dictionary<string, object>
                {
                    ["side"] = "csharp",
                    ["iters"] = Iters,
                    ["runtime"] = System.Runtime.InteropServices.RuntimeInformation
                                      .FrameworkDescription,
                },
            },
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"wrote {outPath}");
        return 0;
    }

    private static double Model(double t, double[] p)
    {
        double z1 = (t - p[1]) / p[2];
        double z2 = (t - p[4]) / p[5];
        double z3 = (t - p[7]) / p[8];
        return p[0] * Math.Exp(-0.5 * z1 * z1)
             + p[3] * Math.Exp(-0.5 * z2 * z2)
             + p[6] * Math.Exp(-0.5 * z3 * z3);
    }

    private static Vector<double> Residual(Vector<double> t, Vector<double> y,
                                           double[] p)
    {
        var r = Vector<double>.Build.Dense(t.Count);
        for (int i = 0; i < t.Count; i++)
        {
            r[i] = Model(t[i], p) - y[i];
        }
        return r;
    }

    private static Matrix<double> FdJacobian(Vector<double> t, Vector<double> y,
                                             double[] p, int m)
    {
        var J = Matrix<double>.Build.Dense(m, N);
        var f0 = Residual(t, y, p);
        var pp = (double[])p.Clone();
        for (int k = 0; k < N; k++)
        {
            double h = 1e-8 * Math.Max(1.0, Math.Abs(p[k]));
            Array.Copy(p, pp, N);
            pp[k] += h;
            var fk = Residual(t, y, pp);
            for (int i = 0; i < m; i++)
            {
                J[i, k] = (fk[i] - f0[i]) / h;
            }
        }
        return J;
    }

    /// <summary>Right-multiply J by diag(d) — what <c>Common.RightMultiply</c>
    /// does to the Jacobian each iteration.</summary>
    private static Matrix<double> ScaleColumns(Matrix<double> J, Vector<double> d)
    {
        int m = J.RowCount;
        var B = Matrix<double>.Build.Dense(m, N);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < N; j++)
            {
                B[i, j] = J[i, j] * d[j];
            }
        }

        return B;
    }

    private static Matrix<double> Augmented(Matrix<double> J, int m)
    {
        var A = Matrix<double>.Build.Dense(m + N, N);
        A.SetSubMatrix(0, 0, J);
        return A;
    }

    private static double Best(Action fn, int iters)
    {
        var times = new List<double>(iters);
        for (int i = 0; i < iters; i++)
        {
            var sw = Stopwatch.StartNew();
            fn();
            sw.Stop();
            times.Add(sw.Elapsed.TotalSeconds);
        }
        times.Sort();
        return times[0];
    }

    private static double NextNormal(Random rng)
    {
        // Box-Muller, enough for synthetic noise
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
