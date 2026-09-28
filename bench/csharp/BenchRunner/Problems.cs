using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp.Bench;

/// <summary>
/// C# mirror of the benchmark problems defined in <c>bench/run_python.py</c>.
/// The two implementations must agree element-by-element.
/// </summary>
internal static class Problems
{
    private const double Sqrt5 = 2.2360679774997896964091736687313;
    private const double Sqrt10 = 3.1622776601683793319988935444327;
    private const double Sqrt90 = 9.4868329805051379959966806192986;

    public static Func<Vector<double>, Vector<double>> Fun(
        string name, Vector<double>[] data, int[] shape)
        => Make(name, data, shape).Fun;

    public static Func<Vector<double>, Matrix<double>>? Jac(
        string name, Vector<double>[] data, int[] shape)
        => Make(name, data, shape).Jac;

    private static (Func<Vector<double>, Vector<double>> Fun,
                    Func<Vector<double>, Matrix<double>>? Jac) Make(
        string name, Vector<double>[] data, int[] shape)
    {
        switch (name)
        {
            case "rosenbrock_n2":
                return (
                    x => Vector<double>.Build.DenseOfArray(new[]
                    {
                        10.0 * (x[1] - x[0] * x[0]),
                        1.0 - x[0],
                    }),
                    x => Matrix<double>.Build.DenseOfRowArrays(
                        new[] { -20.0 * x[0], 10.0 },
                        new[] { -1.0, 0.0 }));

            case "beale_n2":
                return (
                    x => Vector<double>.Build.DenseOfArray(new[]
                    {
                        1.5 - x[0] * (1 - x[1]),
                        2.25 - x[0] * (1 - x[1] * x[1]),
                        2.625 - x[0] * (1 - x[1] * x[1] * x[1]),
                    }),
                    x => Matrix<double>.Build.DenseOfRowArrays(
                        new[] { -(1 - x[1]), x[0] },
                        new[] { -(1 - x[1] * x[1]), 2 * x[0] * x[1] },
                        new[] { -(1 - x[1] * x[1] * x[1]), 3 * x[0] * x[1] * x[1] }));

            case "powell_singular_n4":
                return (
                    x => Vector<double>.Build.DenseOfArray(new[]
                    {
                        x[0] + 10 * x[1],
                        Sqrt5 * (x[2] - x[3]),
                        (x[1] - 2 * x[2]) * (x[1] - 2 * x[2]),
                        Sqrt10 * (x[0] - x[3]) * (x[0] - x[3]),
                    }),
                    x => Matrix<double>.Build.DenseOfRowArrays(
                        new[] { 1.0, 10.0, 0.0, 0.0 },
                        new[] { 0.0, 0.0, Sqrt5, -Sqrt5 },
                        new[] { 0.0, 2 * (x[1] - 2 * x[2]), -4 * (x[1] - 2 * x[2]), 0.0 },
                        new[] { 2 * Sqrt10 * (x[0] - x[3]), 0.0, 0.0, -2 * Sqrt10 * (x[0] - x[3]) }));

            case "wood_n4":
                return (
                    x => Vector<double>.Build.DenseOfArray(new[]
                    {
                        10 * (x[1] - x[0] * x[0]),
                        1 - x[0],
                        Sqrt90 * (x[3] - x[2] * x[2]),
                        1 - x[2],
                        Sqrt10 * (x[1] + x[3] - 2),
                        (x[1] - x[3]) / Sqrt10,
                    }),
                    x => Matrix<double>.Build.DenseOfRowArrays(
                        new[] { -20 * x[0], 10.0, 0.0, 0.0 },
                        new[] { -1.0, 0.0, 0.0, 0.0 },
                        new[] { 0.0, 0.0, -2 * Sqrt90 * x[2], Sqrt90 },
                        new[] { 0.0, 0.0, -1.0, 0.0 },
                        new[] { 0.0, Sqrt10, 0.0, Sqrt10 },
                        new[] { 0.0, 1 / Sqrt10, 0.0, -1 / Sqrt10 }));

            case "box3d_n3":
            {
                var t = Vector<double>.Build.Dense(10, i => 0.1 * (i + 1));
                return (
                    x => Vector<double>.Build.Dense(10, i =>
                        Math.Exp(-t[i] * x[0]) - Math.Exp(-t[i] * x[1])
                        - x[2] * (Math.Exp(-t[i]) - Math.Exp(-10 * t[i]))),
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(10, 3);
                        for (int i = 0; i < 10; i++)
                        {
                            j[i, 0] = -t[i] * Math.Exp(-t[i] * x[0]);
                            j[i, 1] = t[i] * Math.Exp(-t[i] * x[1]);
                            j[i, 2] = -(Math.Exp(-t[i]) - Math.Exp(-10 * t[i]));
                        }
                        return j;
                    });
            }

            case var s when s.StartsWith("dbv_n", StringComparison.Ordinal):
            {
                int n = int.Parse(s["dbv_n".Length..]);
                double h = 1.0 / (n + 1);
                var t = Vector<double>.Build.Dense(n, i => h + (1 - 2 * h) * i / (n - 1));
                return (
                    x =>
                    {
                        var f = Vector<double>.Build.Dense(n);
                        for (int i = 0; i < n; i++)
                        {
                            double prev = i == 0 ? 0.0 : x[i - 1];
                            double next = i == n - 1 ? 0.0 : x[i + 1];
                            double term = x[i] + t[i] + 1;
                            f[i] = 2 * x[i] - prev - next + h * h * term * term * term / 2;
                        }
                        return f;
                    },
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(n, n);
                        for (int i = 0; i < n; i++)
                        {
                            double term = x[i] + t[i] + 1;
                            j[i, i] = 2 + 1.5 * h * h * term * term;
                            if (i > 0)
                            {
                                j[i, i - 1] = -1.0;
                            }
                            if (i < n - 1)
                            {
                                j[i, i + 1] = -1.0;
                            }
                        }
                        return j;
                    });
            }

            case var s when s.StartsWith("broyden_n", StringComparison.Ordinal):
            {
                int n = int.Parse(s["broyden_n".Length..]);
                return (
                    x =>
                    {
                        var f = Vector<double>.Build.Dense(n, i => (3 - 2 * x[i]) * x[i] + 1);
                        for (int i = 1; i < n; i++)
                        {
                            f[i] -= x[i - 1];
                        }
                        for (int i = 0; i < n - 1; i++)
                        {
                            f[i] -= 2 * x[i + 1];
                        }
                        return f;
                    },
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(n, n);
                        for (int i = 0; i < n; i++)
                        {
                            j[i, i] = 3 - 4 * x[i];
                            if (i > 0)
                            {
                                j[i, i - 1] = -1.0;
                            }
                            if (i < n - 1)
                            {
                                j[i, i + 1] = -2.0;
                            }
                        }
                        return j;
                    });
            }

            case "linear_rank_deficient":
            case "underdetermined_m6_n12":
            case "linear_illcond_n40":
            {
                var a = Dense(data[0], shape[0], shape[1]);
                var b = data[1];
                return (x => a * x - b, _ => a);
            }

            case var s when s.StartsWith("sinbig_n", StringComparison.Ordinal):
            {
                var a = Dense(data[0], shape[0], shape[1]);
                var b = data[1];
                int m = a.RowCount;
                int n = a.ColumnCount;
                return (
                    x => a * x.PointwiseSin() - b,
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(m, n);
                        for (int c = 0; c < n; c++)
                        {
                            double cs = Math.Cos(x[c]);
                            for (int r = 0; r < m; r++)
                            {
                                j[r, c] = a[r, c] * cs;
                            }
                        }
                        return j;
                    });
            }

            case "exp_fit_m80":
            case "exp_fit_outliers_m80":
            {
                var t = data[0];
                var y = data[1];
                return (
                    x => Vector<double>.Build.Dense(t.Count,
                        i => x[0] + x[1] * Math.Exp(x[2] * t[i]) - y[i]),
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(t.Count, 3);
                        for (int i = 0; i < t.Count; i++)
                        {
                            double e = Math.Exp(x[2] * t[i]);
                            j[i, 0] = 1.0;
                            j[i, 1] = e;
                            j[i, 2] = x[1] * t[i] * e;
                        }
                        return j;
                    });
            }

            case "gauss2_m200":
            {
                var t = data[0];
                var y = data[1];
                return (
                    x => Vector<double>.Build.Dense(t.Count, i =>
                    {
                        double u1 = (t[i] - x[1]) / x[2];
                        double u2 = (t[i] - x[4]) / x[5];
                        return x[0] * Math.Exp(-0.5 * u1 * u1)
                             + x[3] * Math.Exp(-0.5 * u2 * u2) - y[i];
                    }),
                    null);
            }

            case "damped_osc_m150":
            {
                var t = data[0];
                var y = data[1];
                return (
                    x => Vector<double>.Build.Dense(t.Count, i =>
                        x[0] * Math.Exp(-x[1] * t[i]) * Math.Sin(x[2] * t[i] + x[3])
                        + x[4] - y[i]),
                    null);
            }

            case string g when g.StartsWith("gauss3", StringComparison.Ordinal)
                            || g.StartsWith("mscan_gauss3", StringComparison.Ordinal):
            {
                // p = [A1, mu1, s1, A2, mu2, s2, A3, mu3, s3]   n = 9
                var t = data[0];
                var y = data[1];
                return (
                    x => Vector<double>.Build.Dense(t.Count, i =>
                    {
                        double z1 = (t[i] - x[1]) / x[2];
                        double z2 = (t[i] - x[4]) / x[5];
                        double z3 = (t[i] - x[7]) / x[8];
                        double g1 = x[0] * Math.Exp(-0.5 * z1 * z1);
                        double g2 = x[3] * Math.Exp(-0.5 * z2 * z2);
                        double g3 = x[6] * Math.Exp(-0.5 * z3 * z3);
                        return (g1 + g2 + g3) - y[i];
                    }),
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(t.Count, 9);
                        for (int k = 0; k < 3; k++)
                        {
                            double a = x[3 * k], mu = x[3 * k + 1], sg = x[3 * k + 2];
                            for (int i = 0; i < t.Count; i++)
                            {
                                double z = (t[i] - mu) / sg;
                                double e = Math.Exp(-0.5 * z * z);
                                j[i, 3 * k] = e;
                                j[i, 3 * k + 1] = a * e * z / sg;
                                j[i, 3 * k + 2] = a * e * z * z / sg;
                            }
                        }
                        return j;
                    });
            }

            case "decay_peak_n9":
            {
                // p = [A1, tau1, A2, tau2, Ag, mu, sigma, c0, c1]   n = 9
                var t = data[0];
                var y = data[1];
                return (
                    x => Vector<double>.Build.Dense(t.Count, i =>
                    {
                        double g1 = x[0] * Math.Exp(-t[i] / x[1]);
                        double g2 = x[2] * Math.Exp(-t[i] / x[3]);
                        double z = (t[i] - x[5]) / x[6];
                        double g3 = x[4] * Math.Exp(-0.5 * z * z);
                        double lin = x[7] + x[8] * t[i];
                        return (g1 + g2 + g3 + lin) - y[i];
                    }),
                    x =>
                    {
                        var j = Matrix<double>.Build.Dense(t.Count, 9);
                        for (int i = 0; i < t.Count; i++)
                        {
                            double e1 = Math.Exp(-t[i] / x[1]);
                            j[i, 0] = e1;
                            j[i, 1] = x[0] * e1 * t[i] / (x[1] * x[1]);
                            double e2 = Math.Exp(-t[i] / x[3]);
                            j[i, 2] = e2;
                            j[i, 3] = x[2] * e2 * t[i] / (x[3] * x[3]);
                            double z = (t[i] - x[5]) / x[6];
                            double eg = Math.Exp(-0.5 * z * z);
                            j[i, 4] = eg;
                            j[i, 5] = x[4] * eg * z / x[6];
                            j[i, 6] = x[4] * eg * z * z / x[6];
                            j[i, 7] = 1.0;
                            j[i, 8] = t[i];
                        }
                        return j;
                    });
            }

            default:
                throw new ArgumentException($"unknown problem: {name}");
        }
    }

    /// <summary>Rebuilds a dense row-major matrix from a base64 float64 blob.</summary>
    private static Matrix<double> Dense(Vector<double> flat, int rows, int cols)
        => Matrix<double>.Build.Dense(rows, cols, (i, j) => flat[i * cols + j]);
}
