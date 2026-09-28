// Nine-parameter curve_fit cross-check against scipy.optimize.curve_fit.
//
// Reads bench/problems.json (same file the Python side reads) and fits the two
// nine-parameter models with the public CurveFit.Fit API -- the entry point a
// real caller uses.  Emits popt / pcov diagonal / perr and a best-of-N wall
// clock time.

using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp;

namespace TRF.Sharp.Bench.CurveFit9;

internal static class Program
{
    private const int Reps = 9;

    private static readonly string Root = FindBenchRoot();

    private static int Main(string[] args)
    {
        string outPath = args.Length > 0 ? args[0] : "results_curvefit9.json";
        var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Root, "problems.json")));
        var problems = doc.RootElement.GetProperty("problems");

        string[] wanted = { "gauss3_n9", "decay_peak_n9", "gauss3_outliers_n9" };
        var cases = new List<Dictionary<string, object>>();

        foreach (var prob in problems.EnumerateArray())
        {
            string name = prob.GetProperty("name").GetString()!;
            if (!wanted.Contains(name))
            {
                continue;
            }

            var t = Decode(prob.GetProperty("data").GetProperty("t"));
            var y = Decode(prob.GetProperty("data").GetProperty("y"));
            var x0 = prob.GetProperty("x0").EnumerateArray().Select(ToDouble).ToArray();

            // Bounds: reuse the bounds of the first bounded config of the same
            // problem family so both sides constrain identically.
            double[]? lb = null, ub = null;
            foreach (var conf in prob.GetProperty("configs").EnumerateArray())
            {
                double[]? clb = Arr(conf, "lb"), cub = Arr(conf, "ub");
                if (clb is not null && cub is not null)
                {
                    lb = clb;
                    ub = cub;
                    break;
                }
            }

            Func<double, double[], double> modelArr =
                name.StartsWith("gauss3", StringComparison.Ordinal) ? Gauss3 : DecayPeak;
            Func<double, Vector<double>, double> modelVec =
                (tt, p) => modelArr(tt, Buff(p));

            foreach (bool bounded in new[] { false, true })
            {
                // Two overloads are timed: the convenience one takes
                // double[] and allocates a fresh array per data point, the
                // Vector<double> one does not.
                foreach (bool useVector in new[] { false, true })
                {
                    string suffix = (bounded ? "bounded" : "unbounded")
                                    + (useVector ? "/vector" : "/array");

                    // warm-up (also validates the call)
                    _ = Run(modelArr, modelVec, t, y, x0, lb, ub, bounded, useVector);

                    var times = new List<double>(Reps);
                    CurveFitResult fit = null!;
                    for (int r = 0; r < Reps; r++)
                    {
                        var sw = Stopwatch.StartNew();
                        fit = Run(modelArr, modelVec, t, y, x0, lb, ub, bounded,
                                  useVector);
                        sw.Stop();
                        times.Add(sw.Elapsed.TotalSeconds);
                    }
                    times.Sort();

                    cases.Add(new Dictionary<string, object>
                    {
                        ["case"] = $"{name}/{suffix}",
                        ["problem"] = name,
                        ["bounded"] = bounded,
                        ["overload"] = useVector ? "vector" : "array",
                        ["n"] = x0.Length,
                        ["m"] = t.Count,
                        ["popt"] = fit.Popt.ToArray(),
                        ["pcov_diag"] = Enumerable.Range(0, x0.Length)
                            .Select(i => fit.Pcov[i, i]).ToArray(),
                        ["perr"] = fit.Perp.ToArray(),
                        ["perr_norm"] = fit.Perp.L2Norm(),
                        ["covariance_unreliable"] = fit.CovarianceUnreliable,
                        ["has_active_bounds"] = fit.HasActiveBounds,
                        ["cost"] = fit.Result.Cost,
                        ["nfev"] = fit.Result.Nfev,
                        ["status"] = fit.Result.Status,
                        ["t_min"] = times[0],
                        ["reps"] = Reps,
                    });
                    Console.WriteLine(
                        $"  {name}/{suffix,-20} cost={fit.Result.Cost:E6} " +
                        $"nfev={fit.Result.Nfev,3} status={fit.Result.Status} " +
                        $"t={times[0] * 1e3,8:F3} ms");
                }
            }
        }

        var outDoc = new Dictionary<string, object>
        {
            ["cases"] = cases,
            ["meta"] = new Dictionary<string, object>
            {
                ["reps"] = Reps,
                ["note"] = "TRF.Sharp CurveFit.Fit, nine-parameter models",
            },
        };
        File.WriteAllText(outPath, JsonSerializer.Serialize(
            outDoc, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"wrote {outPath}: {cases.Count} cases");
        return 0;
    }

    [ThreadStatic] private static double[]? s_buf;

    private static double[] Buff(Vector<double> p)
    {
        s_buf ??= new double[p.Count];
        for (int i = 0; i < p.Count; i++)
        {
            s_buf[i] = p[i];
        }
        return s_buf;
    }

    private static CurveFitResult Run(
        Func<double, double[], double> modelArr,
        Func<double, Vector<double>, double> modelVec,
        Vector<double> t, Vector<double> y, double[] x0,
        double[]? lb, double[]? ub, bool bounded, bool useVector)
        => useVector
            ? CurveFit.Fit(modelVec, t, y, Vector<double>.Build.DenseOfArray(x0),
                           lowerBounds: bounded ? lb : null,
                           upperBounds: bounded ? ub : null)
            : CurveFit.Fit(modelArr, t.ToArray(), y.ToArray(), (double[])x0.Clone(),
                           lowerBounds: bounded ? lb : null,
                           upperBounds: bounded ? ub : null);

    private static double Gauss3(double t, double[] p)
    {
        double z1 = (t - p[1]) / p[2];
        double z2 = (t - p[4]) / p[5];
        double z3 = (t - p[7]) / p[8];
        return p[0] * Math.Exp(-0.5 * z1 * z1)
             + p[3] * Math.Exp(-0.5 * z2 * z2)
             + p[6] * Math.Exp(-0.5 * z3 * z3);
    }

    private static double DecayPeak(double t, double[] p)
    {
        double z = (t - p[5]) / p[6];
        return p[0] * Math.Exp(-t / p[1])
             + p[2] * Math.Exp(-t / p[3])
             + p[4] * Math.Exp(-0.5 * z * z)
             + p[7] + p[8] * t;
    }

    private static Vector<double> Decode(JsonElement el)
    {
        var bytes = Convert.FromBase64String(el.GetProperty("b64").GetString()!);
        var values = new double[bytes.Length / 8];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        if (!BitConverter.IsLittleEndian)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = BitConverter.Int64BitsToDouble(
                    System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(
                        BitConverter.DoubleToInt64Bits(values[i])));
            }
        }
        return Vector<double>.Build.DenseOfArray(values);
    }

    private static double[]? Arr(JsonElement el, string name)
    {
        if (!el.TryGetProperty(name, out var v) || v.ValueKind == JsonValueKind.Null)
        {
            return null;
        }
        return v.EnumerateArray().Select(ToDouble).ToArray();
    }

    private static double ToDouble(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => e.GetString() == "inf"
            ? double.PositiveInfinity
            : double.NegativeInfinity,
        _ => e.GetDouble(),
    };

    private static string FindBenchRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "problems.json")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        throw new FileNotFoundException("could not locate bench/problems.json");
    }
}
