// Does a thin-QR + 9x9 SVD reproduce MathNet's full SVD of the augmented
// Jacobian, and is it faster?
//
// The trust-region subproblem needs only the singular values `s` and the right
// singular vectors `V` (uf is already obtained as (V^T g_h)/s in TRF.Sharp),
// so the expensive left factor U is never needed:
//
//     A = Q R        (thin QR,  R is n x n upper triangular)
//     R = U_R S V^T  ->  A = (Q U_R) S V^T
//
// hence s(A) = s(R) and V(A) = V(R).  U is never formed.
//
// Three factorisations are timed:
//   full   : MathNet A.Svd()                       (today's code path)
//   mnqr   : MathNet A.QR() + SVD(R)               (Gram-Schmidt, allocates Q)
//   hhqr   : TRF.Sharp.Core.ThinQr.FactorR + SVD(R) (Householder, R only)

using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp.Core;

namespace TRF.Sharp.Bench.SvdProbe;

internal static class Program
{
    private const int N = 9;
    private static readonly int[] Ms =
        { 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096 };

    // per-column decay: 0.35 -> cond ~6e2, 0.8 -> cond ~2e6
    private static readonly double[] Decays = { 0.35, 0.8 };

    private static void Main()
    {
        var rows = new List<Dictionary<string, object>>();
        foreach (double decay in Decays)
        {
            Console.WriteLine();
            Console.WriteLine($"--- column decay {decay} "
                              + $"(cond ~ {Math.Pow(10.0, decay * (N - 1)):E0}) ---");
            Console.WriteLine($"{"m",6} {"full",10} {"mnQR",10} {"hhQR",10} "
                              + $"{"hh/full",8} {"err s",10} {"err |V|",10} "
                              + $"{"err GN step",12}");
            foreach (int m in Ms)
            {
                var row = Run(m, decay);
                rows.Add(row);
                Console.WriteLine(
                    $"{m,6} {Us(row["svd_full"]),10} {Us(row["svd_mnqr"]),10} "
                    + $"{Us(row["svd_hhqr"]),10} {row["hh_speedup"],8:F2}x "
                    + $"{row["err_s"],10:E2} {row["err_abs_v"],10:E2} "
                    + $"{row["err_gn_step"],12:E2}");
            }
        }

        File.WriteAllText("results_svdprobe.json",
            System.Text.Json.JsonSerializer.Serialize(
                new Dictionary<string, object>
                {
                    ["rows"] = rows,
                    ["meta"] = new Dictionary<string, object>
                    {
                        ["note"] = "MathNet full SVD vs MathNet QR vs "
                                   + "Householder-R-only QR + SVD(R); n = 9",
                    },
                },
                new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine("\nwrote results_svdprobe.json");
    }

    private static string Us(object t) => $"{(double)t * 1e6,8:F1}us";

    private static Dictionary<string, object> Run(int m, double decay)
    {
        var rng = new Random(4242 + m);
        var A = Matrix<double>.Build.Dense(m + N, N);
        for (int i = 0; i < m + N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i, j] = NextNormal(rng) * Math.Pow(10.0, -j * decay);
            }
        }

        var full = A.Svd();
        var sFull = full.S;
        var vFull = full.VT.Transpose();

        // ---- MathNet QR path ----
        var qr = A.QR();
        var svdMn = qr.R.SubMatrix(0, N, 0, N).Svd();
        var sMn = svdMn.S;
        var vMn = svdMn.VT.Transpose();

        // ---- Householder R-only path ----
        var svdHh = ThinQr.FactorR(A).Svd();
        var sHh = svdHh.S;
        var vHh = svdHh.VT.Transpose();

        (double sErr, double vAbsErr) = CompareFactors(sFull, vFull, sHh, vHh);
        (double sErrMn, double vAbsErrMn) = CompareFactors(sFull, vFull, sMn, vMn);

        // ---- end-to-end: does the Gauss-Newton step change? ----
        // p = -V (suf / s^2)  with suf = V^T g_h ; a sign flip in a column
        // of V flips the matching entry of suf, so the product is invariant.
        var rng2 = new Random(99 + m);
        // gh stands in for the scaled gradient g_h = d .* g
        var gh = Vector<double>.Build.Dense(N, _ => NextNormal(rng2));

        var pFull = GnStep(sFull, vFull, gh);
        var pHh = GnStep(sHh, vHh, gh);

        double pErr = 0;
        for (int i = 0; i < N; i++)
        {
            pErr = Math.Max(pErr, Math.Abs(pHh[i] - pFull[i])
                                  / Math.Max(1e-12, Math.Abs(pFull[i])));
        }

        int iters = m <= 128 ? 300 : m <= 512 ? 100 : 20;
        double tFull = Best(() => { _ = A.Svd(); }, iters);
        double tMn = Best(() =>
        {
            var q = A.QR();
            _ = q.R.SubMatrix(0, N, 0, N).Svd();
        }, iters);
        double tHh = Best(() => { _ = ThinQr.FactorR(A).Svd(); }, iters);

        return new Dictionary<string, object>
        {
            ["m"] = m,
            ["decay"] = decay,
            ["svd_full"] = tFull,
            ["svd_mnqr"] = tMn,
            ["svd_hhqr"] = tHh,
            ["hh_speedup"] = tFull / tHh,
            ["mn_speedup"] = tFull / tMn,
            ["err_s"] = sErr,
            ["err_s_mnqr"] = sErrMn,
            ["err_abs_v"] = vAbsErr,
            ["err_abs_v_mnqr"] = vAbsErrMn,
            ["err_gn_step"] = pErr,
        };
    }

    private static Vector<double> GnStep(
        Vector<double> s, Matrix<double> v, Vector<double> gh)
    {
        var suf = v.TransposeThisAndMultiply(gh);
        return -(v * suf.PointwiseDivide(s.PointwiseMultiply(s)));
    }

    private static (double sErr, double vAbsErr) CompareFactors(
        Vector<double> sFull, Matrix<double> vFull,
        Vector<double> sFast, Matrix<double> vFast)
    {
        double sErr = 0, vAbsErr = 0;
        for (int i = 0; i < N; i++)
        {
            sErr = Math.Max(sErr, Math.Abs(sFast[i] - sFull[i]) / Math.Abs(sFull[i]));
            for (int j = 0; j < N; j++)
            {
                // magnitude-only: if this is ~1e-15 the difference is a
                // pure sign flip (singular vectors are sign-indefinite)
                vAbsErr = Math.Max(vAbsErr,
                    Math.Abs(Math.Abs(vFast[i, j]) - Math.Abs(vFull[i, j]))
                    / Math.Max(1e-12, Math.Abs(vFull[i, j])));
            }
        }

        return (sErr, vAbsErr);
    }

    private static double Best(Action fn, int iters)
    {
        for (int i = 0; i < Math.Max(5, iters / 10); i++)
        {
            fn();       // warm-up
        }

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
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
