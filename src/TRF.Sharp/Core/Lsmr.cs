using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp.Core;

/// <summary>Outcome of one <see cref="Lsmr.Solve"/> run.</summary>
internal sealed class LsmrResult
{
    public Vector<double> X { get; init; } = Vector<double>.Build.Dense(0);

    public int Istop { get; init; }

    public int Iterations { get; init; }

    public double Normr { get; init; }

    public double Normar { get; init; }

    public double NormA { get; init; }

    public double CondA { get; init; }

    public double Normx { get; init; }
}

/// <summary>
/// Port of <c>scipy.sparse.linalg.lsmr</c> (Fong &amp; Saunders, SIAM J. Sci.
/// Comput. 33 (2011) 2950–2971).  Used by TRF when <c>tr_solver = 'lsmr'</c>.
/// </summary>
internal static class Lsmr
{
    public static LsmrResult Solve(
        ILinearOperator A,
        Vector<double> b,
        double damp = 0.0,
        double atol = 1e-6,
        double btol = 1e-6,
        double conlim = 1e8,
        int? maxiter = null,
        bool show = false,
        Vector<double>? x0 = null)
    {
        int m = A.Rows;
        int n = A.Columns;
        int minDim = Math.Min(m, n);
        int maxIter = maxiter ?? minDim;

        var u = b.Clone();
        double normb = b.L2Norm();

        Vector<double> x;
        double beta;
        if (x0 is null)
        {
            x = Vector<double>.Build.Dense(n);
            beta = normb;
        }
        else
        {
            x = x0.Clone();
            u = u - A.MatVec(x);
            beta = u.L2Norm();
        }

        var v = Vector<double>.Build.Dense(n);
        double alpha = 0.0;
        if (beta > 0)
        {
            u = u * (1.0 / beta);
            v = A.RMatVec(u);
            alpha = v.L2Norm();
        }

        if (alpha > 0)
        {
            v = v * (1.0 / alpha);
        }

        // Initialize variables for 1st iteration.
        int itn = 0;
        double zetabar = alpha * beta;
        double alphabar = alpha;
        double rho = 1;
        double rhobar = 1;
        double cbar = 1;
        double sbar = 0;

        var h = v.Clone();
        var hbar = Vector<double>.Build.Dense(n);

        // Initialize variables for estimation of ||r||.
        double betadd = beta;
        double betad = 0;
        double rhodold = 1;
        double tautildeold = 0;
        double thetatilde = 0;
        double zeta = 0;
        double d = 0;

        // Initialize variables for estimation of ||A|| and cond(A).
        double normA2 = alpha * alpha;
        double maxrbar = 0;
        double minrbar = 1e100;
        double normA = Math.Sqrt(normA2);
        double condA = 1;
        double normx = 0;

        int istop = 0;
        double ctol = 0;
        if (conlim > 0)
        {
            ctol = 1.0 / conlim;
        }
        double normr = beta;
        double normar = alpha * beta;

        if (normar == 0)
        {
            return new LsmrResult
            {
                X = x, Istop = istop, Iterations = itn, Normr = normr,
                Normar = normar, NormA = normA, CondA = condA, Normx = normx,
            };
        }

        if (normb == 0)
        {
            x = Vector<double>.Build.Dense(n);
            return new LsmrResult
            {
                X = x, Istop = istop, Iterations = itn, Normr = normr,
                Normar = normar, NormA = normA, CondA = condA, Normx = normx,
            };
        }

        while (itn < maxIter)
        {
            itn += 1;

            u = (u * -alpha) + A.MatVec(v);
            beta = u.L2Norm();

            if (beta > 0)
            {
                u = u * (1.0 / beta);
                v = (v * -beta) + A.RMatVec(u);
                alpha = v.L2Norm();
                if (alpha > 0)
                {
                    v = v * (1.0 / alpha);
                }
            }

            var (chat, shat, alphahat) = SymOrtho(alphabar, damp);

            double rhoold = rho;
            var (c, s, rhoNew) = SymOrtho(alphahat, beta);
            rho = rhoNew;
            double thetanew = s * alpha;
            alphabar = c * alpha;

            double rhobarold = rhobar;
            double zetaold = zeta;
            double thetabar = sbar * rho;
            double rhotemp = cbar * rho;
            var (cbarNew, sbarNew, rhobarNew) = SymOrtho(cbar * rho, thetanew);
            cbar = cbarNew;
            sbar = sbarNew;
            rhobar = rhobarNew;
            zeta = cbar * zetabar;
            zetabar = -sbar * zetabar;

            hbar = (hbar * -(thetabar * rho / (rhoold * rhobarold))) + h;
            x = x + (hbar * (zeta / (rho * rhobar)));
            h = (h * -(thetanew / rho)) + v;

            // Estimate of ||r||.
            double betaacute = chat * betadd;
            double betacheck = -shat * betadd;

            double betahat = c * betaacute;
            betadd = -s * betaacute;

            double thetatildeold = thetatilde;
            var (ctildeold, stildeold, rhotildeold) = SymOrtho(rhodold, thetabar);
            thetatilde = stildeold * rhobar;
            rhodold = ctildeold * rhobar;
            betad = (-stildeold * betad) + (ctildeold * betahat);

            tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold;
            double taud = (zeta - thetatilde * tautildeold) / rhodold;
            d += betacheck * betacheck;
            normr = Math.Sqrt(d + ((betad - taud) * (betad - taud)) + (betadd * betadd));

            // Estimate ||A||.
            normA2 += beta * beta;
            normA = Math.Sqrt(normA2);
            normA2 += alpha * alpha;

            // Estimate cond(A).
            maxrbar = Math.Max(maxrbar, rhobarold);
            if (itn > 1)
            {
                minrbar = Math.Min(minrbar, rhobarold);
            }
            condA = Math.Max(maxrbar, rhotemp) / Math.Min(minrbar, rhotemp);

            // Test for convergence.
            normar = Math.Abs(zetabar);
            normx = x.L2Norm();

            double test1 = normr / normb;
            double test2 = (normA * normr) != 0
                ? normar / (normA * normr)
                : double.PositiveInfinity;
            double test3 = 1.0 / condA;
            double t1 = test1 / (1 + normA * normx / normb);
            double rtol = btol + atol * normA * normx / normb;

            if (itn >= maxIter)
            {
                istop = 7;
            }
            if (1 + test3 <= 1)
            {
                istop = 6;
            }
            if (1 + test2 <= 1)
            {
                istop = 5;
            }
            if (1 + t1 <= 1)
            {
                istop = 4;
            }
            if (test3 <= ctol)
            {
                istop = 3;
            }
            if (test2 <= atol)
            {
                istop = 2;
            }
            if (test1 <= rtol)
            {
                istop = 1;
            }

            if (istop > 0)
            {
                break;
            }
        }

        return new LsmrResult
        {
            X = x, Istop = istop, Iterations = itn, Normr = normr,
            Normar = normar, NormA = normA, CondA = condA, Normx = normx,
        };
    }

    /// <summary>Port of <c>scipy.sparse.linalg._isolve.lsqr._sym_ortho</c>.</summary>
    private static (double c, double s, double r) SymOrtho(double a, double b)
    {
        if (b == 0.0)
        {
            return (Math.Sign(a), 0.0, Math.Abs(a));
        }

        if (a == 0.0)
        {
            return (0.0, Math.Sign(b), Math.Abs(b));
        }

        if (Math.Abs(b) > Math.Abs(a))
        {
            double tau = a / b;
            double s = Math.Sign(b) / Math.Sqrt(1 + tau * tau);
            double c = s * tau;
            double r = b / s;
            return (c, s, r);
        }
        else
        {
            double tau = b / a;
            double c = Math.Sign(a) / Math.Sqrt(1 + tau * tau);
            double s = c * tau;
            double r = a / c;
            return (c, s, r);
        }
    }
}
