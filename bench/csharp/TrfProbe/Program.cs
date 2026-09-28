using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;
using TRF.Sharp.Core;

// Reproduces one trust-region iteration of Box-3D (bounded) at the iterate
// where the C# run and the SciPy run diverge, and prints every intermediate
// quantity so it can be diffed against bench/probe_py.json.
internal static class Probe
{
    private static readonly Vector<double> T =
        Vector<double>.Build.Dense(10, i => 0.1 * (i + 1));

    private static Vector<double> Fun(Vector<double> p) =>
        Vector<double>.Build.Dense(10, i =>
            Math.Exp(-T[i] * p[0]) - Math.Exp(-T[i] * p[1])
            - p[2] * (Math.Exp(-T[i]) - Math.Exp(-10 * T[i])));

    private static Matrix<double> Jac(Vector<double> p)
    {
        var j = Matrix<double>.Build.Dense(10, 3);
        for (int i = 0; i < 10; i++)
        {
            j[i, 0] = -T[i] * Math.Exp(-T[i] * p[0]);
            j[i, 1] = T[i] * Math.Exp(-T[i] * p[1]);
            j[i, 2] = -(Math.Exp(-T[i]) - Math.Exp(-10 * T[i]));
        }
        return j;
    }

    private static string Fmt(IEnumerable<double> v) =>
        "[" + string.Join(", ", v.Select(t => t.ToString("R"))) + "]";

    private static void P(string label, Vector<double> v) =>
        Console.WriteLine($"{label,-8}= {Fmt(v)}");

    public static void Main()
    {
        var doc = JsonDocument.Parse(File.ReadAllText("probe_py.json"));
        var x = Vector<double>.Build.DenseOfArray(
            doc.RootElement.GetProperty("x").EnumerateArray()
                .Select(e => e.GetDouble()).ToArray());

        var lb = Vector<double>.Build.DenseOfArray(new[] { -1.0, 0.0, 0.0 });
        var ub = Vector<double>.Build.Dense(3, double.PositiveInfinity);

        var f = Fun(x);
        var J = Jac(x);
        var g = Common.ComputeGrad(J, f);
        var (v, dv) = Common.ClScalingVector(x, g, lb, ub);
        var d = v.Map(Math.Sqrt);
        var diagH = g.PointwiseMultiply(dv);

        int m = J.RowCount, n = J.ColumnCount;
        var jAug = Matrix<double>.Build.Dense(m + n, n);
        jAug.SetSubMatrix(0, 0, Common.RightMultiply(J, d));
        for (int i = 0; i < n; i++)
        {
            jAug[m + i, i] = Math.Sqrt(Math.Max(0.0, diagH[i]));
        }

        var svd = jAug.Svd();
        var fAug = Vector<double>.Build.Dense(m + n);
        for (int i = 0; i < m; i++)
        {
            fAug[i] = f[i];
        }
        var uf = svd.U.TransposeThisAndMultiply(fAug);

        var pGn = Vector<double>.Build.Dense(n);
        for (int i = 0; i < n; i++)
        {
            pGn -= (uf[i] / svd.S[i]) * svd.VT.Row(i);
        }

        P("x", x);
        P("g", g);
        P("v", v);
        P("dv", dv);
        P("d", d);
        P("diag_h", diagH);
        P("sv", svd.S);
        P("uf", uf);
        P("p_gn", pGn);
        Console.WriteLine($"||p_gn|| = {pGn.L2Norm():R}");

        int k = svd.S.Count;
        var Ue = svd.U.SubMatrix(0, m + n, 0, k);
        for (int i = 0; i < k; i++)
        {
            Console.WriteLine($"  U[:,{i}] = " +
                string.Join(", ", Ue.Column(i).Select(t => t.ToString("E6"))));
        }
        for (int i = 0; i < k; i++)
        {
            var vi = svd.VT.Row(i);
            Console.WriteLine($"  V[{i}]   = " +
                string.Join(", ", vi.Select(t => t.ToString("E6"))));
            Console.WriteLine($"    ||Ja@V[{i}]|| = {(jAug * vi).L2Norm():E6}  (s={svd.S[i]:E6})");
        }
        Console.WriteLine("  uf_economy = " + Fmt(Ue.TransposeThisAndMultiply(fAug)));
        Console.WriteLine("Ja:");
        for (int i = 0; i < m + n; i++)
        {
            Console.WriteLine("  " + Fmt(jAug.Row(i).ToArray()));
        }
    }
}
