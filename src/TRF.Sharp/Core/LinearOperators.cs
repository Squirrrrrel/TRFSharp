using MathNet.Numerics.LinearAlgebra;

namespace TRF.Sharp.Core;

/// <summary>
/// Minimal equivalent of <c>scipy.sparse.linalg.LinearOperator</c>: a matrix
/// known only through its action on vectors (and that of its transpose).
/// </summary>
public interface ILinearOperator
{
    /// <summary>Number of rows.</summary>
    int Rows { get; }

    /// <summary>Number of columns.</summary>
    int Columns { get; }

    /// <summary>Computes A·x.</summary>
    Vector<double> MatVec(Vector<double> x);

    /// <summary>Computes Aᵀ·y.</summary>
    Vector<double> RMatVec(Vector<double> y);

    /// <summary>Computes A·X column by column.</summary>
    Matrix<double> MatMat(Matrix<double> X)
    {
        var result = Matrix<double>.Build.Dense(Rows, X.ColumnCount);
        for (int j = 0; j < X.ColumnCount; j++)
        {
            result.SetColumn(j, MatVec(X.Column(j)));
        }
        return result;
    }
}

/// <summary>Adapter exposing a dense <see cref="Matrix{T}"/> as an <see cref="ILinearOperator"/>.</summary>
public sealed class DenseLinearOperator : ILinearOperator
{
    private readonly Matrix<double> _matrix;

    public DenseLinearOperator(Matrix<double> matrix)
    {
        _matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
    }

    public int Rows => _matrix.RowCount;

    public int Columns => _matrix.ColumnCount;

    public Vector<double> MatVec(Vector<double> x) => _matrix.Multiply(x);

    public Vector<double> RMatVec(Vector<double> y) => _matrix.TransposeThisAndMultiply(y);

    public Matrix<double> MatMat(Matrix<double> X) => _matrix.Multiply(X);
}

/// <summary>
/// Equivalent of <c>common.right_multiplied_operator</c>: represents J·diag(d)
/// without forming the product.
/// </summary>
internal sealed class RightMultipliedOperator : ILinearOperator
{
    private readonly ILinearOperator _j;
    private readonly Vector<double> _d;

    public RightMultipliedOperator(ILinearOperator j, Vector<double> d)
    {
        _j = j;
        _d = d;
    }

    public int Rows => _j.Rows;

    public int Columns => _j.Columns;

    public Vector<double> MatVec(Vector<double> x) => _j.MatVec(x.PointwiseMultiply(_d));

    public Vector<double> RMatVec(Vector<double> y) => _d.PointwiseMultiply(_j.RMatVec(y));
}

/// <summary>
/// Equivalent of <c>common.regularized_lsq_operator</c>: the vertically stacked
/// operator [ J ; diag(diag) ] with shape (m + n, n).
/// </summary>
internal sealed class RegularizedLsqOperator : ILinearOperator
{
    private readonly ILinearOperator _j;
    private readonly Vector<double> _diag;

    public RegularizedLsqOperator(ILinearOperator j, Vector<double> diag)
    {
        _j = j;
        _diag = diag;
    }

    public int Rows => _j.Rows + _j.Columns;

    public int Columns => _j.Columns;

    public Vector<double> MatVec(Vector<double> x)
    {
        var top = _j.MatVec(x);
        var result = Vector<double>.Build.Dense(Rows);
        result.SetSubVector(0, top.Count, top);
        for (int i = 0; i < _diag.Count; i++)
        {
            result[_j.Rows + i] = _diag[i] * x[i];
        }
        return result;
    }

    public Vector<double> RMatVec(Vector<double> y)
    {
        var y1 = Vector<double>.Build.DenseOfVector(y.SubVector(0, _j.Rows));
        var y2 = y.SubVector(_j.Rows, _diag.Count);
        return _j.RMatVec(y1) + _diag.PointwiseMultiply(y2);
    }
}
