using System.Text.Json;
using System.Text.Json.Serialization;
using MathNet.Numerics.LinearAlgebra;
using Xunit;

namespace TRF.Sharp.Tests;

/// <summary>
/// Compares TRF.Sharp against reference results produced by SciPy on this
/// machine (see tests/reference/gen_reference.py).
/// </summary>
public class ReferenceTests
{
    private static readonly ReferenceData Data = ReferenceData.Load();

    public static IEnumerable<object[]> LeastSquaresCaseNames =>
        Data.LeastSquares.Select(c => new object[] { c.Name });

    public static IEnumerable<object[]> CurveFitCaseNames =>
        Data.CurveFit.Select(c => new object[] { c.Name });

    [Theory]
    [MemberData(nameof(LeastSquaresCaseNames))]
    public void LeastSquaresMatchesScipy(string name)
    {
        var expectedCase = Data.LeastSquares.First(c => c.Name == name);
        var expected = expectedCase.Expected;
        var actual = Problems.Run(name);

        var errors = new List<string>();

        for (int i = 0; i < expected.X.Length; i++)
        {
            if (!Close(actual.X[i], expected.X[i]))
            {
                errors.Add($"x[{i}]: got {actual.X[i]:R}, expected {expected.X[i]:R}");
            }
        }

        if (!Close(actual.Cost, expected.Cost, scaleReference: true))
        {
            errors.Add($"cost: got {actual.Cost:R}, expected {expected.Cost:R}");
        }

        if (actual.Nfev != expected.Nfev)
        {
            errors.Add($"nfev: got {actual.Nfev}, expected {expected.Nfev}");
        }

        if (actual.Njev != expected.Njev)
        {
            errors.Add($"njev: got {actual.Njev}, expected {expected.Njev}");
        }

        if (actual.Status != expected.Status)
        {
            errors.Add($"status: got {actual.Status}, expected {expected.Status}");
        }

        for (int i = 0; i < expected.ActiveMask.Length; i++)
        {
            if (actual.ActiveMask[i] != expected.ActiveMask[i])
            {
                errors.Add($"active_mask[{i}]: got {actual.ActiveMask[i]}, expected {expected.ActiveMask[i]}");
            }
        }

        Assert.True(
            errors.Count == 0,
            $"{name} deviates from SciPy:\n  " + string.Join("\n  ", errors));
    }

    [Theory]
    [MemberData(nameof(CurveFitCaseNames))]
    public void CurveFitMatchesScipy(string name)
    {
        var expected = Data.CurveFit.First(c => c.Name == name);
        var actual = Problems.RunCurveFit(name);

        var errors = new List<string>();
        for (int i = 0; i < expected.Popt.Length; i++)
        {
            if (!Close(actual.Popt[i], expected.Popt[i]))
            {
                errors.Add($"popt[{i}]: got {actual.Popt[i]:R}, expected {expected.Popt[i]:R}");
            }
        }

        for (int i = 0; i < expected.Popt.Length; i++)
        {
            double expectedVar = expected.Pcov[i][i];
            if (!Close(actual.Pcov[i, i], expectedVar, scaleReference: true))
            {
                errors.Add($"pcov[{i},{i}]: got {actual.Pcov[i, i]:R}, expected {expectedVar:R}");
            }
        }

        Assert.True(
            errors.Count == 0,
            $"curve_fit/{name} deviates from SciPy:\n  " + string.Join("\n  ", errors));
    }

    [Fact]
    public void OptimalRosenbrockIsReproduced()
    {
        // Values printed in the SciPy documentation for the bounded Rosenbrock.
        var res = LeastSquares.Solve(
            Problems.Rosen,
            Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 }),
            new LeastSquaresOptions
            {
                Jacobian = Problems.RosenJac,
                LowerBounds = new[] { double.NegativeInfinity, 1.5 },
                UpperBounds = new[] { double.PositiveInfinity, double.PositiveInfinity },
            });

        Assert.Equal(1.22437075, res.X[0], 6);
        Assert.Equal(1.5, res.X[1], 6);
        Assert.Equal(0.025213093946805685, res.Cost, 8);
    }

    [Fact]
    public void RosenbrockWithoutBoundsIsReproduced()
    {
        var res = LeastSquares.Solve(
            Problems.Rosen,
            Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 }),
            new LeastSquaresOptions { Jacobian = Problems.RosenJac });

        Assert.Equal(1.0, res.X[0], 8);
        Assert.Equal(1.0, res.X[1], 8);
        Assert.True(res.Cost < 1e-25, $"cost = {res.Cost:R}");
    }

    private static bool Close(double actual, double expected, bool scaleReference = false)
    {
        double scale = Math.Max(1.0, Math.Abs(expected));
        double value = scaleReference ? Math.Max(1.0, Math.Abs(expected)) : scale;
        return Math.Abs(actual - expected) <= 1e-6 * value;
    }
}

/// <summary>The test problems, defined identically to the Python reference script.</summary>
internal static class Problems
{
    public static readonly Vector<double> T = Vector<double>.Build.Dense(15, i => i * (10.0 / 14.0));

    public static readonly Vector<double> YExp =
        Vector<double>.Build.Dense(15, i => 0.5 + 2.0 * Math.Exp(-1.0 * (i * (10.0 / 14.0))));

    public static readonly Vector<double> XLin = Vector<double>.Build.Dense(10, i => i);

    public static readonly Vector<double> YLin = Vector<double>.Build.Dense(10, i => 1.0 - 2.0 * i);

    public static Vector<double> Rosen(Vector<double> x) =>
        Vector<double>.Build.DenseOfArray(new[] { 10 * (x[1] - x[0] * x[0]), 1 - x[0] });

    public static Matrix<double> RosenJac(Vector<double> x) =>
        Matrix<double>.Build.DenseOfRowArrays(
            new[] { -20 * x[0], 10.0 },
            new[] { -1.0, 0.0 });

    public static Vector<double> ExpResiduals(Vector<double> p) =>
        Vector<double>.Build.Dense(15, i => p[0] + p[1] * Math.Exp(p[2] * T[i]) - YExp[i]);

    public static Vector<double> LinResiduals(Vector<double> p) =>
        Vector<double>.Build.Dense(10, i => p[0] * XLin[i] + p[1] - YLin[i]);

    public static Vector<double> RankDeficient(Vector<double> p) =>
        Vector<double>.Build.DenseOfArray(new[] { p[0] + p[1] - 1.0, 2 * p[0] + 2 * p[1] - 2.0 });

    public static Vector<double> Underdetermined(Vector<double> p) =>
        Vector<double>.Build.DenseOfArray(new[] { p[0] * p[1] - 1.0 });

    public static Vector<double> Broyden(Vector<double> p)
    {
        var f = Vector<double>.Build.Dense(p.Count, i => (3 - p[i]) * p[i] + 1);
        for (int i = 1; i < p.Count; i++)
        {
            f[i] -= p[i - 1];
        }
        for (int i = 0; i < p.Count - 1; i++)
        {
            f[i] -= 2 * p[i + 1];
        }
        return f;
    }

    public static LeastSquaresResult Run(string name)
    {
        var x0Two = Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 });
        var lbRosen = new[] { double.NegativeInfinity, 1.5 };
        var ubRosen = new[] { double.PositiveInfinity, double.PositiveInfinity };

        return name switch
        {
            "rosen_analytic_unbounded" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions { Jacobian = RosenJac }),

            "rosen_analytic_bounded" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    Jacobian = RosenJac, LowerBounds = lbRosen, UpperBounds = ubRosen,
                }),

            "rosen_fd2_unbounded" => LeastSquares.Solve(Rosen, x0Two),

            "rosen_fd2_bounded" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions { LowerBounds = lbRosen, UpperBounds = ubRosen }),

            "rosen_fd3_bounded" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    JacobianMethod = JacobianMethod.ThreePoint,
                    LowerBounds = lbRosen,
                    UpperBounds = ubRosen,
                }),

            "exp_fit_interior" => LeastSquares.Solve(
                ExpResiduals,
                Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0, 0.0 }),
                new LeastSquaresOptions
                {
                    LowerBounds = new[] { 0.0, 0.0, -5.0 },
                    UpperBounds = new[] { 10.0, 10.0, 5.0 },
                }),

            "exp_fit_upper_active" => LeastSquares.Solve(
                ExpResiduals,
                Vector<double>.Build.DenseOfArray(new[] { 0.2, 1.0, 0.0 }),
                new LeastSquaresOptions
                {
                    LowerBounds = new[] { 0.0, 0.0, -5.0 },
                    UpperBounds = new[] { 0.4, 10.0, 5.0 },
                }),

            "linear_fit_at_lower_bound" => LeastSquares.Solve(
                LinResiduals,
                Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0 }),
                new LeastSquaresOptions { LowerBounds = new[] { 0.0, double.NegativeInfinity } }),

            "rosen_xscale_jac" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    Jacobian = RosenJac, LowerBounds = lbRosen, UpperBounds = ubRosen, XScaleJac = true,
                }),

            "rosen_xscale_array" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    Jacobian = RosenJac,
                    LowerBounds = lbRosen,
                    UpperBounds = ubRosen,
                    XScale = new[] { 2.0, 0.5 },
                }),

            "exp_fit_soft_l1" => LeastSquares.Solve(
                ExpResiduals,
                Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0, 0.0 }),
                new LeastSquaresOptions
                {
                    LowerBounds = new[] { 0.0, 0.0, -5.0 },
                    UpperBounds = new[] { 10.0, 10.0, 5.0 },
                    Loss = Loss.SoftL1,
                    FScale = 0.1,
                }),

            "exp_fit_huber" => SolveExpWith(Loss.Huber),
            "exp_fit_cauchy" => SolveExpWith(Loss.Cauchy),
            "exp_fit_arctan" => SolveExpWith(Loss.Arctan),

            "rank_deficient" => LeastSquares.Solve(
                RankDeficient,
                Vector<double>.Build.DenseOfArray(new[] { 0.5, 0.5 }),
                new LeastSquaresOptions { LowerBounds = new[] { -1.0, -1.0 }, UpperBounds = new[] { 1.0, 1.0 } }),

            "underdetermined" => LeastSquares.Solve(
                Underdetermined,
                Vector<double>.Build.DenseOfArray(new[] { 1.0, 2.0 }),
                new LeastSquaresOptions { LowerBounds = new[] { 0.1, 0.1 }, UpperBounds = new[] { 5.0, 5.0 } }),

            "rosen_lsmr_bounded" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    Jacobian = RosenJac,
                    LowerBounds = lbRosen,
                    UpperBounds = ubRosen,
                    TrustRegionSolver = TrustRegionSolver.Lsmr,
                }),

            "rosen_lsmr_unbounded" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    Jacobian = RosenJac,
                    TrustRegionSolver = TrustRegionSolver.Lsmr,
                }),

            "exp_fit_x0_on_bounds" => LeastSquares.Solve(
                ExpResiduals,
                Vector<double>.Build.DenseOfArray(new[] { 0.1, 4.0, -3.0 }),
                new LeastSquaresOptions
                {
                    LowerBounds = new[] { 0.1, 0.5, -3.0 },
                    UpperBounds = new[] { 2.0, 4.0, -0.1 },
                }),

            "broyden_fd2_n30" => LeastSquares.Solve(
                Broyden,
                Vector<double>.Build.Dense(30, -1.0)),

            "rosen_fd2_diff_step" => LeastSquares.Solve(
                Rosen, x0Two, new LeastSquaresOptions
                {
                    LowerBounds = lbRosen,
                    UpperBounds = ubRosen,
                    DiffStep = 1e-6,
                }),

            _ => throw new ArgumentException($"Unknown case: {name}"),
        };
    }

    private static LeastSquaresResult SolveExpWith(Loss loss) =>
        LeastSquares.Solve(
            ExpResiduals,
            Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0, 0.0 }),
            new LeastSquaresOptions
            {
                LowerBounds = new[] { 0.0, 0.0, -5.0 },
                UpperBounds = new[] { 10.0, 10.0, 5.0 },
                Loss = loss,
                FScale = 0.1,
            });

    public static CurveFitResult RunCurveFit(string name) => name switch
    {
        "curve_exp" => CurveFit.Fit(
            (t, p) => p[0] + p[1] * Math.Exp(p[2] * t),
            T,
            YExp,
            Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0, 0.0 }),
            new[] { 0.0, 0.0, -5.0 },
            new[] { 10.0, 10.0, 5.0 }),

        "curve_lin_clamped" => CurveFit.Fit(
            (t, p) => p[0] * t + p[1],
            XLin,
            YLin,
            Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0 }),
            new[] { 0.0, double.NegativeInfinity },
            new[] { double.PositiveInfinity, double.PositiveInfinity }),

        _ => throw new ArgumentException($"Unknown curve_fit case: {name}"),
    };
}

internal sealed class ReferenceCase
{
    public string Name { get; set; } = string.Empty;

    public ReferenceExpected Expected { get; set; } = new();
}

internal sealed class ReferenceExpected
{
    public double[] X { get; set; } = Array.Empty<double>();

    public double Cost { get; set; }

    public double Optimality { get; set; }

    public int Nfev { get; set; }

    public int Njev { get; set; }

    public int Status { get; set; }

    public int[] ActiveMask { get; set; } = Array.Empty<int>();
}

internal sealed class ReferenceCurveCase
{
    public string Name { get; set; } = string.Empty;

    public double[] Popt { get; set; } = Array.Empty<double>();

    public double[][] Pcov { get; set; } = Array.Empty<double[]>();

    [JsonPropertyName("pcov_diag")]
    public double[] PcovDiag { get; set; } = Array.Empty<double>();
}

internal sealed class ReferenceData
{
    [JsonPropertyName("scipy_version")]
    public string ScipyVersion { get; set; } = string.Empty;

    [JsonPropertyName("least_squares")]
    public List<ReferenceCase> LeastSquares { get; set; } = new();

    [JsonPropertyName("curve_fit")]
    public List<ReferenceCurveCase> CurveFit { get; set; } = new();

    public static ReferenceData Load()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "reference.json");
        var json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<ReferenceData>(
            json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
    }
}
