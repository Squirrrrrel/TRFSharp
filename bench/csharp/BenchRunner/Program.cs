using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp;

namespace TRF.Sharp.Bench;

internal sealed record TrajectoryPoint(
    [property: JsonPropertyName("x")] double[] X,
    [property: JsonPropertyName("nit")] int Nit,
    [property: JsonPropertyName("nfev")] int Nfev);

internal sealed record CaseResult
{
    [JsonPropertyName("problem")] public string Problem { get; set; } = "";
    [JsonPropertyName("config")] public string Config { get; set; } = "";
    [JsonPropertyName("x")] public double[] X { get; set; } = Array.Empty<double>();
    [JsonPropertyName("cost")] public double Cost { get; set; }
    [JsonPropertyName("optimality")] public double Optimality { get; set; }
    [JsonPropertyName("nfev")] public int Nfev { get; set; }
    [JsonPropertyName("njev")] public int Njev { get; set; }
    [JsonPropertyName("nit")] public int Nit { get; set; }
    [JsonPropertyName("status")] public int Status { get; set; }
    [JsonPropertyName("success")] public bool Success { get; set; }
    [JsonPropertyName("message")] public string Message { get; set; } = "";
    [JsonPropertyName("active_mask")] public int[] ActiveMask { get; set; } = Array.Empty<int>();
    [JsonPropertyName("fun_norm")] public double FunNorm { get; set; }
    [JsonPropertyName("grad_inf")] public double GradInf { get; set; }
    [JsonPropertyName("trajectory")] public List<TrajectoryPoint> Trajectory { get; set; } = new();
    [JsonPropertyName("t_min")] public double TMin { get; set; }
    [JsonPropertyName("t_median")] public double TMedian { get; set; }
    [JsonPropertyName("t_mean")] public double TMean { get; set; }
    [JsonPropertyName("reps")] public int Reps { get; set; }
    [JsonPropertyName("error")] public string? Error { get; set; }
}

internal sealed class BenchMeta
{
    [JsonPropertyName("engine")] public string Engine { get; set; } = "";
    [JsonPropertyName("trf_sharp_version")] public string Version { get; set; } = "";
    [JsonPropertyName("mathnet_version")] public string MathNetVersion { get; set; } = "";
    [JsonPropertyName("runtime")] public string Runtime { get; set; } = "";
    [JsonPropertyName("os")] public string Os { get; set; } = "";
    [JsonPropertyName("arch")] public string Arch { get; set; } = "";
    [JsonPropertyName("build")] public string Build { get; set; } = "";
}

internal static class Program
{
    private const int MaxTraj = 400;

    private static bool s_verbose2;

    public static int Main(string[] args)
    {
        string outPath = args.Length > 0 ? args[0] : "results_csharp.json";
        string? filter = args.Length > 1 ? args[1] : null;   // substring on "name/config"
        s_verbose2 = args.Length > 2 && args[2] == "v2";   // per-iteration trace
        var doc = JsonDocument.Parse(File.ReadAllText("problems.json"));
        var root = doc.RootElement;

        var results = new List<CaseResult>();

        // Global JIT warm-up so the first real case is not penalised.
        WarmUp();

        foreach (var prob in root.GetProperty("problems").EnumerateArray())
        {
            string name = prob.GetProperty("name").GetString()!;
            int n = prob.GetProperty("n").GetInt32();
            int m = prob.GetProperty("m").GetInt32();
            var x0 = prob.GetProperty("x0").EnumerateArray().Select(ToDouble).ToArray();

            var (data, shape) = LoadData(prob);

            foreach (var conf in prob.GetProperty("configs").EnumerateArray())
            {
                string cid = conf.GetProperty("id").GetString()!;
                if (filter is not null && !$"{name}/{cid}".Contains(
                        filter, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                var sw = Stopwatch.StartNew();
                CaseResult rec;
                try
                {
                    rec = Run(name, n, m, x0, data, shape, conf);
                }
                catch (Exception ex)
                {
                    rec = new CaseResult
                    {
                        Problem = name,
                        Config = cid,
                        Error = $"{ex.GetType().Name}: {ex.Message}",
                    };
                }
                sw.Stop();
                results.Add(rec);
                Console.WriteLine(
                    $"{(rec.Error is null ? "    " : "ERR ")}{name}/{cid,-24} {sw.Elapsed.TotalSeconds,8:F3}s");
            }
        }

        var meta = new BenchMeta
        {
            Engine = "TRF.Sharp LeastSquares.Solve (port of scipy least_squares method='trf')",
            Version = typeof(LeastSquares).Assembly.GetName().Version?.ToString() ?? "n/a",
            MathNetVersion = typeof(Matrix<double>).Assembly.GetName().Version?.ToString() ?? "n/a",
            Runtime = RuntimeInformation.FrameworkDescription,
            Os = RuntimeInformation.OSDescription.Trim(),
            Arch = RuntimeInformation.ProcessArchitecture.ToString(),
            Build = typeof(Program).Assembly
                .GetCustomAttribute<AssemblyConfigurationAttribute>()?.Configuration ?? "n/a",
        };

        var json = new JsonSerializerOptions { WriteIndented = false };
        File.WriteAllText(outPath, JsonSerializer.Serialize(
            new Dictionary<string, object> { ["meta"] = meta, ["results"] = results }, json));

        Console.WriteLine($"\nwrote {outPath} ({results.Count} cases)");
        return 0;
    }

    private static void WarmUp()
    {
        for (int i = 0; i < 4; i++)
        {
            LeastSquares.Solve(
                x => Vector<double>.Build.DenseOfArray(new[]
                {
                    10.0 * (x[1] - x[0] * x[0]), 1.0 - x[0],
                }),
                Vector<double>.Build.DenseOfArray(new[] { -2.0, 1.0 }));
        }
    }

    private static (Vector<double>[] Data, int[] Shape) LoadData(JsonElement prob)
    {
        if (!prob.TryGetProperty("data", out var dataEl) || dataEl.ValueKind != JsonValueKind.Object)
        {
            return (Array.Empty<Vector<double>>(), Array.Empty<int>());
        }

        var list = new List<Vector<double>>();
        int[] shape = Array.Empty<int>();
        foreach (var prop in dataEl.EnumerateObject())
        {
            if (prop.Name == "A")
            {
                shape = prop.Value.GetProperty("shape").EnumerateArray()
                    .Select(e => e.GetInt32()).ToArray();
            }
            list.Add(Decode(prop.Value));
        }
        return (list.ToArray(), shape);
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

    /// <summary>Reads a JSON number, or the "inf"/"-inf" sentinels used for
    /// non-finite bounds (JSON has no +/-Infinity literal).</summary>
    private static double ToDouble(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.String => e.GetString() == "inf"
            ? double.PositiveInfinity
            : double.NegativeInfinity,
        _ => e.GetDouble(),
    };

    private static CaseResult Run(
        string name, int n, int m, double[] x0,
        Vector<double>[] data, int[] shape, JsonElement conf)
    {
        string cid = conf.GetProperty("id").GetString()!;
        string jacMode = conf.GetProperty("jac").GetString()!;

        var fun = Problems.Fun(name, data, shape);
        var analytic = Problems.Jac(name, data, shape);

        var opt = new LeastSquaresOptions
        {
            LowerBounds = Arr(conf, "lb"),
            UpperBounds = Arr(conf, "ub"),
            JacobianMethod = jacMode == "fd3" ? JacobianMethod.ThreePoint : JacobianMethod.TwoPoint,
            Loss = (conf.GetProperty("loss").ValueKind == JsonValueKind.Null
                ? "linear"
                : conf.GetProperty("loss").GetString()) switch
            {
                "soft_l1" => Loss.SoftL1,
                "huber" => Loss.Huber,
                "cauchy" => Loss.Cauchy,
                "arctan" => Loss.Arctan,
                _ => Loss.Linear,
            },
            FScale = conf.GetProperty("f_scale").ValueKind == JsonValueKind.Null
                ? 1.0
                : conf.GetProperty("f_scale").GetDouble(),
            TrustRegionSolver = (conf.GetProperty("tr_solver").ValueKind == JsonValueKind.Null
                ? "exact"
                : conf.GetProperty("tr_solver").GetString()) switch
            {
                "lsmr" => TrustRegionSolver.Lsmr,
                _ => TrustRegionSolver.Exact,
            },
        };

        if (conf.GetProperty("ftol").ValueKind != JsonValueKind.Null)
        {
            opt.Ftol = conf.GetProperty("ftol").GetDouble();
        }
        if (conf.GetProperty("xtol").ValueKind != JsonValueKind.Null)
        {
            opt.Xtol = conf.GetProperty("xtol").GetDouble();
        }
        if (conf.GetProperty("gtol").ValueKind != JsonValueKind.Null)
        {
            opt.Gtol = conf.GetProperty("gtol").GetDouble();
        }
        if (conf.GetProperty("diff_step").ValueKind != JsonValueKind.Null)
        {
            opt.DiffStep = conf.GetProperty("diff_step").GetDouble();
        }

        var xs = conf.GetProperty("x_scale");
        if (xs.ValueKind == JsonValueKind.String && xs.GetString() == "jac")
        {
            opt.XScaleJac = true;
        }
        else if (xs.ValueKind == JsonValueKind.Array)
        {
            opt.XScale = xs.EnumerateArray().Select(ToDouble).ToArray();
        }

        if (jacMode == "analytic")
        {
            opt.Jacobian = analytic ?? throw new InvalidOperationException(
                $"{name} has no analytic Jacobian");
        }

        if (s_verbose2)
        {
            opt.Verbose = 2;
        }

        var x0v = Vector<double>.Build.DenseOfArray(x0);
        var traj = new List<TrajectoryPoint>();
        opt.Callback = r =>
        {
            if (traj.Count < MaxTraj)
            {
                traj.Add(new TrajectoryPoint(r.X.ToArray(), r.Nit, r.Nfev));
            }
            return false;
        };

        // warm-up (also validates)
        LeastSquares.Solve(fun, x0v, opt);
        traj.Clear();

        long work = (long)n * m;
        int reps = work <= 4000 ? 9 : work <= 200_000 ? 5 : 3;

        var times = new List<double>(reps);
        LeastSquaresResult res = null!;
        for (int i = 0; i < reps; i++)
        {
            traj.Clear();
            var sw = Stopwatch.StartNew();
            res = LeastSquares.Solve(fun, x0v, opt);
            sw.Stop();
            times.Add(sw.Elapsed.TotalSeconds);
        }

        times.Sort();
        return new CaseResult
        {
            Problem = name,
            Config = cid,
            X = res.X.ToArray(),
            Cost = res.Cost,
            Optimality = res.Optimality,
            Nfev = res.Nfev,
            Njev = res.Njev,
            Nit = res.Nit,
            Status = res.Status,
            Success = res.Success,
            Message = res.Message,
            ActiveMask = res.ActiveMask,
            FunNorm = res.Fun.L2Norm(),
            GradInf = res.Grad.InfinityNorm(),
            Trajectory = new List<TrajectoryPoint>(traj),
            TMin = times[0],
            TMedian = times[times.Count / 2],
            TMean = times.Average(),
            Reps = reps,
            Error = null,
        };
    }
}
