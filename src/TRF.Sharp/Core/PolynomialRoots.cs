using System.Numerics;

namespace TRF.Sharp.Core;

/// <summary>
/// Port of the behaviour of <c>numpy.roots</c> that SciPy relies on in
/// <c>common.solve_trust_region_2d</c>: all roots of a polynomial with real
/// coefficients, given in <b>descending</b> order (leading coefficient first).
/// Leading and trailing zeros are handled the same way NumPy handles them.
/// </summary>
internal static class PolynomialRoots
{
    /// <summary>All (complex) roots of the polynomial.</summary>
    public static Complex[] FindRoots(double[] coefficientsDescending)
    {
        if (coefficientsDescending is null || coefficientsDescending.Length == 0)
        {
            return Array.Empty<Complex>();
        }

        // Trim leading zeros (NumPy does this too).
        int start = 0;
        while (start < coefficientsDescending.Length && coefficientsDescending[start] == 0.0)
        {
            start++;
        }
        if (start == coefficientsDescending.Length)
        {
            return Array.Empty<Complex>();
        }

        int end = coefficientsDescending.Length - 1;
        int zeroRoots = 0;
        while (end > start && coefficientsDescending[end] == 0.0)
        {
            zeroRoots++;
            end--;
        }

        int length = end - start + 1;
        var core = new double[length];
        Array.Copy(coefficientsDescending, start, core, 0, length);

        Complex[] coreRoots = length switch
        {
            1 => Array.Empty<Complex>(),
            2 => new[] { Linear(core[0], core[1]) },
            3 => Quadratic(core[0], core[1], core[2]),
            _ => DurandKerner(core),
        };

        if (zeroRoots == 0)
        {
            return coreRoots;
        }

        var all = new Complex[coreRoots.Length + zeroRoots];
        Array.Copy(coreRoots, all, coreRoots.Length);
        for (int i = coreRoots.Length; i < all.Length; i++)
        {
            all[i] = Complex.Zero;
        }
        return all;
    }

    /// <summary>Only the real roots (numerically real ones).</summary>
    public static double[] FindRealRoots(double[] coefficientsDescending)
    {
        var roots = FindRoots(coefficientsDescending);
        var real = new List<double>();
        foreach (var root in roots)
        {
            if (Math.Abs(root.Imaginary) <= 1e-10 * Math.Max(1.0, Math.Abs(root.Real)))
            {
                real.Add(root.Real);
            }
        }
        return real.ToArray();
    }

    private static Complex Linear(double a, double b) => new(-b / a, 0.0);

    private static Complex[] Quadratic(double a, double b, double c)
    {
        double disc = b * b - 4 * a * c;
        if (disc >= 0)
        {
            double sqrtDisc = Math.Sqrt(disc);
            double q = -0.5 * (b + Math.Sign(b) * sqrtDisc);
            if (q == 0.0)
            {
                return new[] { new Complex(0.0, 0.0), new Complex(0.0, 0.0) };
            }
            return new[] { new Complex(q / a, 0.0), new Complex(c / q, 0.0) };
        }

        double root = Math.Sqrt(-disc);
        return new[]
        {
            new Complex(-b / (2 * a), root / (2 * a)),
            new Complex(-b / (2 * a), -root / (2 * a)),
        };
    }

    /// <summary>
    /// Weierstrass / Durand–Kerner iteration.  Deterministic and adequate for
    /// the low-degree polynomials occurring in the 2-D trust-region subproblem.
    /// </summary>
    private static Complex[] DurandKerner(double[] descending)
    {
        int degree = descending.Length - 1;

        // Convert to monic ascending form: p(x) = x^d + a[d-1] x^(d-1) + ... + a[0]
        var a = new double[degree];
        double lead = descending[0];
        for (int i = 0; i < degree; i++)
        {
            a[i] = descending[descending.Length - 1 - i] / lead;
        }

        var z = new Complex[degree];
        Complex seed = new(0.4, 0.9);
        Complex power = Complex.One;
        for (int k = 0; k < degree; k++)
        {
            z[k] = power;
            power *= seed;
        }

        for (int iteration = 0; iteration < 500; iteration++)
        {
            double maxStep = 0.0;
            for (int k = 0; k < degree; k++)
            {
                Complex value = Evaluate(a, z[k]);
                Complex denominator = Complex.One;
                for (int j = 0; j < degree; j++)
                {
                    if (j == k)
                    {
                        continue;
                    }
                    denominator *= z[k] - z[j];
                }

                if (denominator == Complex.Zero)
                {
                    continue;
                }

                Complex delta = value / denominator;
                z[k] -= delta;
                maxStep = Math.Max(maxStep, delta.Magnitude);
            }

            if (maxStep < 1e-14)
            {
                break;
            }
        }

        return z;
    }

    /// <summary>Horner evaluation of the monic ascending polynomial.</summary>
    private static Complex Evaluate(double[] ascendingMonic, Complex x)
    {
        Complex result = Complex.One; // leading coefficient of the monic form
        for (int i = ascendingMonic.Length - 1; i >= 0; i--)
        {
            result = result * x + ascendingMonic[i];
        }
        return result;
    }
}
