using System.Text;
using MathNet.Numerics.LinearAlgebra;
using Xunit;

namespace TRF.Sharp.Tests;

/// <summary>API-level behaviour: verbosity, callback, validation, convenience overloads.</summary>
public class ApiTests
{
    private static LeastSquaresResult FitRosenbrock(int verbose = 0, Func<LeastSquaresResult, bool>? callback = null) =>
        LeastSquares.Solve(
            Problems.Rosen,
            Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 }),
            new LeastSquaresOptions
            {
                LowerBounds = new[] { double.NegativeInfinity, 1.5 },
                UpperBounds = new[] { double.PositiveInfinity, double.PositiveInfinity },
                Verbose = verbose,
                Callback = callback,
            });

    [Fact]
    public void VerboseTwoPrintsIterationTable()
    {
        var stdout = Console.Out;
        var buffer = new StringWriter();
        try
        {
            Console.SetOut(buffer);
            var result = FitRosenbrock(verbose: 2);
            Assert.True(result.Success);
        }
        finally
        {
            Console.SetOut(stdout);
        }

        var text = buffer.ToString();
        Assert.Contains("Iteration", text);
        Assert.Contains("Optimality", text);
    }

    [Fact]
    public void CallbackCanStopTheOptimizer()
    {
        int calls = 0;
        var result = FitRosenbrock(callback: r =>
        {
            calls++;
            return calls >= 2;
        });

        Assert.Equal(-2, result.Status);
        Assert.False(result.Success);
        Assert.Contains("callback", result.Message);
        Assert.Equal(2, calls);
    }

    [Fact]
    public void CallbackReceivesIntermediateResults()
    {
        var seen = new List<double>();
        FitRosenbrock(callback: r =>
        {
            seen.Add(r.Cost);
            return false;
        });

        Assert.True(seen.Count > 1);
        Assert.True(seen[0] > seen[^1], "cost should decrease along the iterations");
    }

    [Fact]
    public void InvalidBoundsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => LeastSquares.Solve(
            Problems.Rosen,
            Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 }),
            new LeastSquaresOptions { LowerBounds = new[] { 3.0, 0.0 }, UpperBounds = new[] { 1.0, 5.0 } }));
    }

    [Fact]
    public void InitialGuessOutsideBoundsIsRejected()
    {
        Assert.Throws<ArgumentException>(() => LeastSquares.Solve(
            Problems.Rosen,
            Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 }),
            new LeastSquaresOptions { LowerBounds = new[] { -10.0, 3.0 } }));
    }

    [Fact]
    public void WrongJacobianShapeIsRejected()
    {
        Assert.Throws<ArgumentException>(() => LeastSquares.Solve(
            Problems.Rosen,
            Vector<double>.Build.DenseOfArray(new[] { 2.0, 2.0 }),
            new LeastSquaresOptions { Jacobian = _ => Matrix<double>.Build.Dense(3, 3) }));
    }

    [Fact]
    public void NonFiniteInitialResidualsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => LeastSquares.Solve(
            _ => Vector<double>.Build.DenseOfArray(new[] { double.NaN, 1.0 }),
            Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0 })));
    }

    [Fact]
    public void ArrayOverloadMatchesVectorOverload()
    {
        var viaArrays = LeastSquares.Solve(
            x => new[] { 10 * (x[1] - x[0] * x[0]), 1 - x[0] },
            new[] { 2.0, 2.0 },
            new LeastSquaresOptions { LowerBounds = new[] { double.NegativeInfinity, 1.5 } });

        Assert.Equal(1.22437075, viaArrays.X[0], 6);
        Assert.Equal(1.5, viaArrays.X[1], 6);
        Assert.Equal(2, viaArrays.Parameters.Length);
    }

    [Fact]
    public void CurveFitReturnsStandardErrors()
    {
        var fit = CurveFit.Fit(
            (t, p) => p[0] + p[1] * Math.Exp(p[2] * t),
            Problems.T.ToArray(),
            Problems.YExp.ToArray(),
            new[] { 1.0, 1.0, 0.0 },
            new[] { 0.0, 0.0, -5.0 },
            new[] { 10.0, 10.0, 5.0 });

        Assert.Equal(3, fit.Parameters.Length);
        Assert.Equal(0.5, fit.Popt[0], 5);
        Assert.Equal(2.0, fit.Popt[1], 5);
        Assert.Equal(-1.0, fit.Popt[2], 5);
        Assert.False(fit.CovarianceUnreliable);
        Assert.True(fit.Perp[0] > 0 && double.IsFinite(fit.Perp[0]));
    }

    [Fact]
    public void ActiveMaskFlagsClampedParameter()
    {
        var result = LeastSquares.Solve(
            Problems.LinResiduals,
            Vector<double>.Build.DenseOfArray(new[] { 1.0, 1.0 }),
            new LeastSquaresOptions { LowerBounds = new[] { 0.0, double.NegativeInfinity } });

        Assert.Equal(-1, result.ActiveMask[0]);
        Assert.Equal(0, result.ActiveMask[1]);
    }
}
