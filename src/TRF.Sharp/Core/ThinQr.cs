using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp.Core;

/// <summary>
/// Householder thin QR that produces <b>R only</b>.
/// </summary>
/// <remarks>
/// <para>
/// The trust-region subproblem needs just the singular values <c>s</c> and the
/// right singular vectors <c>V</c> of the (augmented) Jacobian — the left factor
/// <c>U</c> is never formed, because <c>uf</c> is obtained as
/// <c>(Vᵀ·g_h)/s</c>. With <c>A = Q·R</c> and <c>R = U_R·S·Vᵀ</c> we get
/// <c>A = (Q·U_R)·S·Vᵀ</c>, so <c>s(A) = s(R)</c> and <c>V(A) = V(R)</c>:
/// the n×n factor <c>R</c> carries everything we need.
/// </para>
/// <para>
/// Computing <c>R</c> costs 2mn² − 2n³/3 flops and never materialises the
/// m×n matrix <c>Q</c>, which is why it beats a full SVD (O(mn²) with a much
/// larger constant, plus an iterative bidiagonalisation) by orders of magnitude
/// once m ≫ n. Householder is used rather than Gram–Schmidt because it is
/// backward stable; the augmented Jacobian is routinely ill conditioned.
/// </para>
/// </remarks>
internal static class ThinQr
{
    /// <summary>
    /// Upper-triangular factor <c>R</c> (n×n) of a thin QR decomposition of
    /// <paramref name="a"/> (m×n, m ≥ n). <c>Q</c> is never formed.
    /// </summary>
    public static Matrix<double> FactorR(Matrix<double> a)
    {
        int m = a.RowCount;
        int n = a.ColumnCount;

        // Row-major scratch copy: the inner loops are contiguous in j.
        var buf = new double[m * n];
        for (int i = 0; i < m; i++)
        {
            int off = i * n;
            for (int j = 0; j < n; j++)
            {
                buf[off + j] = a[i, j];
            }
        }

        var w = new double[n];

        for (int k = 0; k < n; k++)
        {
            double normx = 0.0;
            for (int i = k; i < m; i++)
            {
                double t = buf[i * n + k];
                normx += t * t;
            }

            normx = Math.Sqrt(normx);
            if (normx == 0.0)
            {
                continue;   // column already zero below and on the diagonal
            }

            double x0 = buf[k * n + k];
            // alpha = -sign(x0) * ||x|| : keeps the reflection away from
            // cancellation in beta (see Golub & Van Loan, §5.1.2).
            double alpha = x0 >= 0.0 ? -normx : normx;
            double beta = x0 - alpha;

            // ||v||² with v = x - alpha·e₁
            double vNorm2 = beta * beta + (normx * normx - x0 * x0);
            if (vNorm2 <= 0.0)
            {
                continue;
            }

            // w = (2/||v||²) · A[k:m, k:n]ᵀ · v
            for (int j = k; j < n; j++)
            {
                double sum = beta * buf[k * n + j];
                for (int i = k + 1; i < m; i++)
                {
                    sum += buf[i * n + k] * buf[i * n + j];
                }

                w[j] = 2.0 * sum / vNorm2;
            }

            // A[k:m, k:n] -= v · wᵀ
            for (int i = k; i < m; i++)
            {
                double vi = i == k ? beta : buf[i * n + k];
                if (vi == 0.0)
                {
                    continue;
                }

                int off = i * n;
                for (int j = k; j < n; j++)
                {
                    buf[off + j] -= vi * w[j];
                }
            }

            // The reflected column is (alpha, 0, …, 0) by construction; write it
            // back so the stored triangle is exactly R.
            buf[k * n + k] = alpha;
            for (int i = k + 1; i < m; i++)
            {
                buf[i * n + k] = 0.0;
            }
        }

        var r = Matrix<double>.Build.Dense(n, n);
        for (int i = 0; i < n; i++)
        {
            int off = i * n;
            for (int j = i; j < n; j++)
            {
                r[i, j] = buf[off + j];
            }
        }

        return r;
    }
}
