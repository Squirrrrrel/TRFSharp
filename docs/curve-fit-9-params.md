# 九参数非线性曲线拟合使用指南（TRF.Sharp）

面向目标场景：**9 个参数、几百个数据点**的非线性最小二乘拟合（典型如多峰光谱 /
色谱、多指数衰减 + 背景）。所有代码均来自 `examples/NineParamFit`，可直接编译运行：

```bash
dotnet run -c Release --project examples/NineParamFit
```

---

## 1. 三十秒上手

模型：三个高斯峰叠加，9 个参数
`p = [A1, c1, w1, A2, c2, w2, A3, c3, w3]`（幅值 / 中心 / 宽度 × 3）。

```csharp
using TRF.Sharp;

// 模型函数：单个数据点的预测值
static double Model(double t, double[] p)
{
    double z1 = (t - p[1]) / p[2];
    double z2 = (t - p[4]) / p[5];
    double z3 = (t - p[7]) / p[8];
    return p[0] * Math.Exp(-0.5 * z1 * z1)
         + p[3] * Math.Exp(-0.5 * z2 * z2)
         + p[6] * Math.Exp(-0.5 * z3 * z3);
}

double[] t = ...;   // 256 个横坐标
double[] y = ...;   // 256 个观测值
double[] p0 = { 1.2, 1.35, 0.75, 0.50, 4.25, 1.10, 0.30, 6.55, 0.60 };

double[] lb = { 0.0, 0.0, 0.05, 0.0, 2.0, 0.05, 0.0, 5.0, 0.05 };
double[] ub = { 5.0, 3.0, 2.00, 5.0, 6.0, 2.00, 5.0, 9.0, 2.00 };

var fit = CurveFit.Fit(Model, t, y, p0, lb, ub);

double[] popt = fit.Parameters;   // 拟合结果
double[] perr = fit.Perp.ToArray();   // 每个参数的 1σ 误差
```

实际输出（合成数据，真值 `{1.00, 1.50, 0.55, 0.65, 4.00, 0.90, 0.40, 6.80, 0.45}`）：

```text
popt   1.0004  1.4987  0.5483  0.6485  4.0006  0.9055  0.4002  6.8011  0.4485
perr   0.0009  0.0007  0.0007  0.0007  0.0013  0.0016  0.0010  0.0013  0.0014
cost 1.651E-003  nfev 6  status 1  `gtol` termination condition is satisfied.
```

参数恢复到真值的 ~1e-3，与 `perr`（1σ）自洽。

---

## 2. 两个入口，怎么选

| | `CurveFit.Fit` | `LeastSquares.Solve` |
|---|---|---|
| 对应的 scipy API | `curve_fit(method='trf')` | `least_squares(method='trf')` |
| 输入 | 模型函数 `model(t, p)` + `xdata`/`ydata` | 残差函数 `fun(p) → r`（长度 m） |
| 额外产出 | `Pcov` 协方差、`Perp` 1σ 误差 | 无（只有求解器结果） |
| 适用 | **标准曲线拟合（推荐）** | 自定义残差：加权、差分、多曲线联立等 |

`CurveFit.Fit` 内部就是"把模型包成残差 + 调 `LeastSquares.Solve` + 算协方差"。
如果你的目标只是"给我 popt 和 perr"，用 `CurveFit.Fit`；需要自己构造残差
（例如按区间加权、两条曲线共享参数）时用 `LeastSquares.Solve`。

### API 签名

```csharp
// 数组重载 —— 最顺手
CurveFitResult Fit(
    Func<double, double[], double> model,
    double[] xdata, double[] ydata, double[] p0,
    double[]? lowerBounds = null, double[]? upperBounds = null,
    double[]? sigma = null, bool absoluteSigma = false,
    LeastSquaresOptions? options = null);

// Vector 重载 —— model 直接读 Vector 索引，避免每点分配（见 §6 性能）
CurveFitResult Fit(
    Func<double, Vector<double>, double> model,
    Vector<double> xdata, Vector<double> ydata, Vector<double> p0,
    ...同上...);

// 直接求解
LeastSquaresResult Solve(
    Func<Vector<double>, Vector<double>> fun,   // p → 残差向量（长度 m）
    Vector<double> x0,
    LeastSquaresOptions? options = null);
// 另有 Func<double[], double[]> 的数组重载
```

**残差函数返回的是 `model(p) − y`，不是平方。** 平方和由求解器处理。

---

## 3. 边界与初值

- `lowerBounds` / `upperBounds` 长度必须等于 n（9），或长度为 1（自动广播到所有参数）。
  `null` 表示该侧无界。
- **初值必须落在闭区间 `[lb, ub]` 内**（含端点可以），越界直接抛
  `ArgumentException: Initial guess is outside of provided bounds`。
  这与 scipy 1.17 的行为一致，不会悄悄帮你投影。
- 初值质量决定成败：9 参数问题存在局部极小（峰"认错"位置、两个峰合并到同一个）。
  经验做法：
  - 幅值初值取 `y.max()` 量级，中心取数据里局部极大值的位置，宽度取峰的半高宽量级；
  - 拿不准就先无界拟合一次，把结果钳进边界再作初值；
  - 用 `Callback`（§5）观察迭代轨迹，判断是否掉进错误的极小。
- 结果里某个参数贴边 ≠ 错误，但要看 `HasActiveBounds`：为 true 时协方差只有参考意义。

---

## 4. 结果怎么读

`CurveFitResult`：

| 属性 | 含义 |
|---|---|
| `Popt` / `Parameters` | 拟合参数（Vector / `double[]`） |
| `Pcov` | 参数协方差矩阵（9×9），scipy `curve_fit` 同款算法：JᵀJ 的 SVD 伪逆，默认按残差方差 `cost/(m−n)` 缩放 |
| `Perp` | `sqrt(diag(Pcov))`，每个参数的 1σ 误差 |
| `CovarianceUnreliable` | m ≤ n 时为 true（此时 `Pcov` 填 +∞，同 scipy） |
| `HasActiveBounds` | 有参数贴在边界上（协方差只是参考值） |
| `Result` | 底层 `LeastSquaresResult`（见下） |

`LeastSquaresResult` 的关键字段：`X`、`Cost`（0.5·Σr²，注意不是 Σr²）、`Fun`
（最终残差向量）、`Jac`（最终雅可比）、`Optimality`（一阶最优性度量）、
`Nfev`/`Njev`/`Nit`、`Status`/`Message`、`Success`、`ActiveMask`（每个参数
0 = 自由，−1 = 贴下界，+1 = 贴上界）。

`Status` 终止码与 scipy 相同：`1` gtol（梯度收敛，最常见）、`2` ftol、
`3` xtol、`4` ftol+xtol、`0` 达到 `MaxNfev`（默认 `100·n = 900`）、
`−2` 回调要求停止。**status 0 不一定是失败**，先看 cost 是否已经稳定。

收敛判据默认 `ftol = xtol = gtol = 1e-8`（与 scipy 一致），一般不用动。

---

## 5. 进阶选项（`LeastSquaresOptions`）

```csharp
var options = new LeastSquaresOptions
{
    // —— 雅可比 ——
    Jacobian = JacobianOfResidual,        // Func<Vector, Matrix>：m×n 解析雅可比（推荐）
    JacobianMethod = JacobianMethod.TwoPoint,  // 无解析时的差分：TwoPoint(默认)/ThreePoint
    DiffStep = null,                      // 差分相对步长，null = scipy 的"最优"步长

    // —— 边界（也可通过 Fit 的专用参数传，二选一）——
    LowerBounds = lb,
    UpperBounds = ub,

    // —— 收敛 ——
    Ftol = 1e-8, Xtol = 1e-8, Gtol = 1e-8,
    MaxNfev = null,                       // null = 100·n

    // —— 尺度 ——（二者互斥）
    XScale = new double[] { 1, 1, 0.1, ... },  // 每个参数的特征尺度
    XScaleJac = false,                    // true = x_scale='jac'，自动按雅可比列范数缩放

    // —— 鲁棒损失（有离群点时）——
    Loss = Loss.SoftL1,                   // Linear/SoftL1/Huber/Cauchy/Arctan/Custom
    FScale = 0.02,                        // 内点/离群点的软边界，取残差噪声量级

    // —— 求解器 ——
    TrustRegionSolver = TrustRegionSolver.Auto,  // Auto=Exact，密阵默认即是

    // —— 调试 ——
    Verbose = 0,                          // 0 静默 / 1 终止报告 / 2 逐迭代
    Callback = r => { /* 每次迭代后调用；返回 true 停止 */ return false; },
};
```

要点：

- **`XScale` / `XScaleJac`**：9 个参数量纲差异大（幅值 ~1、宽度 ~0.1、中心 ~5）
  时建议 `XScaleJac = true`，对病态问题（峰重叠）有帮助；代价是每步多做一次
  雅可比列范数计算，本场景开销可忽略。
- **鲁棒损失**：`Loss` 一旦不是 `Linear`，`Cost` 就不再是 0.5·Σr²，`Pcov`/`Perp`
  的统计解释随之失效（scipy 同样如此）——此时把 `Perp` 当作相对不确定度看。
  `FScale` 应取"正常残差"的典型大小（如噪声 σ），而不是 1。
- **`TrustRegionSolver.Lsmr`** 只在 m 上万或雅可比稀疏时才考虑；本场景（m 几百、
  n=9）用默认 `Exact`，别动。

### 五种损失在有离群点数据上的实测（8% 离群点，`FScale=0.02`）

```text
linear   cost 1.466E-001  A2 0.6487  c2 4.0006  nfev 7
soft_l1  cost 4.009E-002  A2 0.6485  c2 4.0010  nfev 11
huber    cost 4.347E-002  A2 0.6485  c2 4.0009  nfev 12
cauchy   cost 1.544E-002  A2 0.6484  c2 4.0007  nfev 14
```

参数都抗住了离群点（真值 A2=0.65, c2=4.00），但 `cauchy` 的 cost 受污染最小。
若你的数据有坏点，`soft_l1` 或 `cauchy` 是好起点。

---

## 6. 解析雅可比：值得写，也容易写错

m = 256 时，解析雅可比把纯求解时间从 ~3.6 ms 降到 ~2.0 ms，且差分在噪声大的
数据上偶有收敛路径差异。写法：

```csharp
// d(残差)/dp，m×9 矩阵：行 = 数据点，列 = 参数
// 注意：如果用了 sigma，这里必须同样除以 sigma —— 雅可比必须对应
// "传给求解器的那个残差向量"
static Matrix<double> Jacobian(Vector<double> p)
{
    var J = Matrix<double>.Build.Dense(M, 9);
    for (int i = 0; i < M; i++)
    {
        double t = T[i];
        for (int k = 0; k < 3; k++)
        {
            double z = (t - p[3*k + 1]) / p[3*k + 2];
            double g = Math.Exp(-0.5 * z * z);
            J[i, 3*k]     = g;                          // d/dA
            J[i, 3*k + 1] = p[3*k] * g * z / p[3*k + 2];       // d/dc
            J[i, 3*k + 2] = p[3*k] * g * z * z / p[3*k + 2];   // d/dw
        }
    }
    return J;
}

var options = new LeastSquaresOptions { Jacobian = Jacobian };
```

校验雅可比最简单的办法：同一问题分别用解析和 `TwoPoint` 跑一遍，比较 `Popt`。
两者应在 1e-6 内一致；不一致就是解析式写错了（本库的对拍基准里，
解析雅可比的良态用例与 scipy 一致到 5.5e-17）。

---

## 7. 性能（9 参数，m = 几百）

实测（Release，best-of-N，Adreno X1-85 / .NET 10）：

| 配置 | m=128 | m=256 | m=512 | m=1024 |
|---|---:|---:|---:|---:|
| 纯求解 + 解析雅可比 | 0.33 ms | 0.66 ms | 2.03 ms | 3.68 ms |
| 同问题 SciPy | 0.76 ms | 0.76 ms | 1.10 ms | 1.50 ms |
| 纯求解 + 有限差分 | 0.52 ms | 1.03 ms | 2.76 ms | 4.94 ms |

（m=32 时更快，~0.15 ms。数据点几百个的完整拟合在 1–5 ms 量级，
即使放进交互式 UI 里也毫无压力。）

三个影响速度的写法细节：

1. **用 `Vector` 重载，别用数组重载**。数组重载每评估一次模型就
   `p.ToArray()` 分配一次（m × nfev 次小分配）；Vector 重载直接索引 `p[i]`。
2. **模型函数里别每点建数组/委托**。逐点 `Func` 调用本身没问题，但在循环里
   new 数组会显著拖慢。
3. **`CurveFit.Fit` = 求解 + 协方差**。协方差是最终雅可比上的一次 SVD
   （m=256 约 1 ms）。实测示例：完整 `Fit` 3.16 ms，其中纯求解 2.04 ms。
   循环里只关心 popt 时，用 `LeastSquares.Solve` 即可省掉这一步。

---

## 8. 与 scipy 的一致性

移植以 `scipy.optimize._lsq`（1.17.1）为基准逐行对拍验证，116 个求解用例 +
25 个九参数专项用例：

- 68/116 与 scipy **逐位一致或 1e-12 内一致**（含整条迭代轨迹）；
- 唯一的 MISMATCH 是一个病态线性问题 + LSMR 的混沌用例（C# 反而 cost 更低），
  与本场景无关；
- 九参数用例 0 MISMATCH：良态问题参数一致到 **5.5e-17**；
  `CurveFit.Fit` 对 `scipy.curve_fit`：popt 一致到 8e-12、`perr` 一致到 4e-12、
  nfev 全等；
- 少数重叠峰用例为 EQUIV：cost 一致到 1e-13 以下但参数沿近平坦方向漂移 ——
  这是问题本身有近不唯一的极小点，不是移植缺陷。遇到时加边界约束或改初值。

对拍注意事项（如果你自己做交叉验证）：

- `scipy.optimize.curve_fit` **无界时默认走 MINPACK `lm` 而不是 TRF**，
  必须显式 `method='trf'` 才是同类比较；
- scipy ≥ 1.16 的结果里没有 `nit`，要用 callback 计数；
- `bounds=(lb, None)` 里的 `None` 会变成 NaN，必须显式传 `±inf`。

---

## 9. 常见坑速查

| 症状 | 原因 / 处理 |
|---|---|
| `Initial guess is outside of provided bounds` | 初值越界。钳进 `[lb, ub]` 内再调（含端点可以）。 |
| `Each lower bound must be strictly less than...` | 某个 `lb[i] >= ub[i]`。 |
| 结果贴边、`ActiveMask` 非 0 | 参数被边界顶住。检查边界是否合理；`HasActiveBounds=true` 时 `Perp` 仅供参考。 |
| 收敛到明显错误的峰位 | 局部极小。改初值（见 §3），或用 `XScaleJac` 稳定病态问题。 |
| status=0 | 达到 `MaxNfev`。先看 cost 是否已稳定；不行就调大 `MaxNfev` 或改初值。 |
| 用了 `Loss` 后 `Perp` 变大很多 | 正常：robust loss 下 `Pcov` 不再是标准统计意义，见 §5。 |
| 数组重载明显慢于 Vector 重载 | 每点 `p.ToArray()` 分配，见 §6 性能第 1 条。 |
| 差分雅可比拟合结果与解析不一致 | 检查解析雅可比（§6 的校验方法）；或噪声太大时改用 `ThreePoint`。 |

---

## 10. 完整示例

`examples/NineParamFit/Program.cs` 覆盖了四种典型调用：

1. `CurveFit.Fit` + 有限差分 —— 最少代码；
2. `CurveFit.Fit` + 解析雅可比 —— 推荐，含协方差开销的拆分计时；
3. 同一数据注入 8% 离群点后四种鲁棒损失的对比；
4. `LeastSquares.Solve` + 自定义加权残差 —— 自由度最大。

```bash
dotnet run -c Release --project examples/NineParamFit
```

完整输出：

```text
nine-parameter three-peak fit, m = 256 data points, n = 9 parameters

[1] CurveFit.Fit, finite-difference Jacobian
    popt   1.0004  1.4987  0.5483  0.6485  4.0006  0.9055  0.4002  6.8011  0.4485
    perr   0.0009  0.0007  0.0007  0.0007  0.0013  0.0016  0.0010  0.0013  0.0014
    cost 1.651E-003  nfev 6  status 1  `gtol` termination condition is satisfied.
    time   3.57 ms

[2] CurveFit.Fit, analytic Jacobian (recommended)
    popt   1.0004  1.4987  0.5483  0.6485  4.0006  0.9055  0.4002  6.8011  0.4485
    perr   0.0009  0.0007  0.0007  0.0007  0.0013  0.0016  0.0010  0.0013  0.0014
    cost 1.651E-003  nfev 6  status 1  `gtol` termination condition is satisfied.
    time   3.16 ms  (solve only: 2.04 ms)

[3] Same data with 8% outliers — robust loss
    linear   cost 1.466E-001  A2 0.6487  c2 4.0006  nfev 7
    soft_l1  cost 4.009E-002  A2 0.6485  c2 4.0010  nfev 11
    huber    cost 4.347E-002  A2 0.6485  c2 4.0009  nfev 12
    cauchy   cost 1.544E-002  A2 0.6484  c2 4.0007  nfev 14

[4] LeastSquares.Solve directly (custom residual)
    x      1.0004  1.4987  0.5483  0.6483  4.0007  0.9055  0.4002  6.8011  0.4485
    cost 9.922E-004  optimality 3.74E-009  nit 5  status 1
```

把三峰模型换成你的模型只需两步：改 `Model`（和可选的 `Jacobian`），改 `p0` 与
边界。其余照旧。

---

*文档对应的库版本：TRF.Sharp 1.0.0（thin-QR 因子分解版）；bench 数据见
`bench/report_nine_param.md` 与 `bench/report.md`。*
