
# 深度学习中的优化算法笔记

## 1. 优化与深度学习

深度学习的训练过程本质上是一个**优化问题**。我们希望通过迭代更新参数 $\theta$ 来最小化一个损失函数 $L(\theta)$。

在统计学习中有两个常见的概念：

* **经验风险（Empirical Risk）**：在有限训练数据上的平均损失

  $$
  \hat R(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(y_i, f(x_i;\theta))
  $$

* **真实风险（Risk）**：在整个数据分布上的期望损失

  $$
  R(\theta) = \mathbb{E}_{(x,y)\sim P}\big[\ell(y, f(x;\theta))\big]
  $$

优化算法的目标是**减少训练误差**，但深度学习的最终目标是**减少泛化误差**。因此，优化和泛化之间有一定差距——即便优化器能让训练误差很低，也不代表模型能在新数据上表现良好。

### 优化中的挑战

1. **局部极小值**：非凸函数可能有很多局部极小点，优化器可能卡在这些点附近。
2. **鞍点**：梯度为零但不是极小值，这会使得更新停滞。
3. **梯度消失/爆炸**：在深层网络中，梯度可能变得极小或极大，导致训练困难。

---

## 2. 凸性（Convexity）

在数学优化中，凸性是一个非常重要的性质。

* **凸集**：集合 $\mathcal{X}$ 中任意两点 $a, b$ 的连线仍在集合中：

  $$
  \lambda a + (1-\lambda)b \in \mathcal{X}, \quad \forall \lambda \in [0,1]
  $$

* **凸函数**：定义在凸集上的函数 $f$ 满足：

  $$
  f(\lambda x + (1-\lambda)x') \leq \lambda f(x) + (1-\lambda)f(x')
  $$

直观理解：凸函数的图像是“碗状”的，没有局部陷阱。

* **詹森不等式**：对随机变量 $X$ 和凸函数 $f$：

  $$
  \mathbb{E}[f(X)] \geq f(\mathbb{E}[X])
  $$

  说明“函数的平均值 ≥ 平均值的函数”。

在深度学习中，大多数优化问题是非凸的，但在局部区域中常常表现出“近似凸性”，这也是各种优化算法能发挥作用的原因。

---

## 3. 梯度下降（Gradient Descent, GD）

### 一维推导

设 $f:\mathbb{R}\to\mathbb{R}$，利用泰勒展开：

$$
f(x+\epsilon) \approx f(x) + \epsilon f'(x)
$$

如果取更新方向为**负梯度方向**：$\epsilon=-\eta f'(x)$，得到：

$$
f(x-\eta f'(x)) \approx f(x) - \eta (f'(x))^2
$$

可见函数值下降了。

更新公式：

$$
x \leftarrow x - \eta f'(x)
$$

### 多维情况

对于 $f:\mathbb{R}^d\to\mathbb{R}$：

$$
\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)
$$

### 学习率的影响

* 学习率 $\eta$ 太小 → 收敛缓慢
* 学习率 $\eta$ 太大 → 参数更新过头，可能发散
* 合适的 $\eta$ 依赖于 Hessian 矩阵的特征值范围（条件数）。

---

## 4. 随机梯度下降（SGD）

在深度学习中，目标函数通常是所有样本损失的平均：

$$
f(\theta) = \frac{1}{n}\sum_{i=1}^n f_i(\theta)
$$

计算全量梯度 $\nabla f(\theta)$ 需要遍历所有样本，代价高。

**SGD思想**：每次只随机采样一个样本 $i$：

$$
\theta \leftarrow \theta - \eta \nabla f_i(\theta)
$$

* 每次更新开销从 $O(n)$ 降到 $O(1)$
* 梯度是无偏估计，即 $\mathbb{E}[\nabla f_i(\theta)] = \nabla f(\theta)$
* 缺点：轨迹嘈杂，容易震荡

为此，引入**学习率衰减**（分段常数 / 指数衰减 / 多项式衰减）来平衡早期快速收敛和后期稳定性。

---

## 5. 小批量 SGD（Mini-batch SGD）

介于 GD 和 SGD 之间：

$$
\theta \leftarrow \theta - \eta \frac{1}{|\mathcal{B}|} \sum_{i\in \mathcal{B}} \nabla f_i(\theta)
$$

其中 $\mathcal{B}$ 是一个小批量（如 32/64/128）。

* **优点**：

  * 利用 GPU 并行，计算效率高
  * 梯度比单样本更稳定
* **缺点**：需要调节合适的 batch size

因此，小批量 SGD 成为深度学习的默认训练方式。

---

## 6. 动量法（Momentum）

在非凸问题中，SGD 会产生“之字形”震荡，特别是在损失函数呈峡谷状时。

**动量法的思想**：累积历史梯度，形成“惯性”。

公式：

$$
v_t = \beta v_{t-1} + \nabla f(\theta_t) \\
\theta_{t+1} = \theta_t - \eta v_t
$$

其中 $\beta$ 通常取 $0.9$。

* 当梯度方向一致时 → 累积加速收敛
* 当梯度方向震荡时 → 平滑更新，减少抖动

动量法被广泛应用于深度学习，几乎所有现代优化器都包含它的思想。

---

## 7. AdaGrad

AdaGrad 针对 **稀疏特征问题** 设计。

### 思想

为不同坐标设置不同的学习率：

* 出现频繁的特征 → 学习率逐渐变小
* 稀疏特征 → 学习率保持较大

公式：

$$
s_t = s_{t-1} + g_t^2 \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t}+\epsilon} g_t
$$

* **优点**：对稀疏数据效果好（如 NLP）
* **缺点**：学习率单调递减，后期可能过小，导致停滞

---

## 8. RMSProp

为了解决 AdaGrad 的衰减过快问题，RMSProp 引入了**指数加权平均**：

$$
s_t = \gamma s_{t-1} + (1-\gamma) g_t^2 \\
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t
$$

特点：

* 学习率不会衰减到接近 0
* 对非凸问题更有效
* 常用参数：$\gamma=0.9,\ \eta=0.001$

---

## 9. Adadelta

Adadelta 是 AdaGrad 的进一步改进：

* 不再依赖全局学习率 $\eta$
* 使用**参数自身的更新历史**作为尺度

核心公式：

* 梯度平方平均：

  $$
  s_t = \rho s_{t-1} + (1-\rho) g_t^2
  $$
* 更新幅度平方平均：

  $$
  \Delta x_t = \rho \Delta x_{t-1} + (1-\rho)(g_t')^2
  $$
* 调整后的梯度：

  $$
  g_t' = \frac{\sqrt{\Delta x_{t-1}+\epsilon}}{\sqrt{s_t+\epsilon}} g_t
  $$
* 参数更新：

  $$
  \theta_{t+1} = \theta_t - g_t'
  $$

特点：不需要显式学习率，完全自适应。

---

## 10. Adam

Adam（Adaptive Moment Estimation）可以看作 **Momentum + RMSProp** 的结合。

* 一阶矩估计（动量）：

  $$
  v_t = \beta_1 v_{t-1} + (1-\beta_1) g_t
  $$
* 二阶矩估计（方差）：

  $$
  s_t = \beta_2 s_{t-1} + (1-\beta_2) g_t^2
  $$
* 偏差修正：

  $$
  \hat v_t = \frac{v_t}{1-\beta_1^t}, \quad \hat s_t = \frac{s_t}{1-\beta_2^t}
  $$
* 参数更新：

  $$
  \theta_{t+1} = \theta_t - \frac{\eta \hat v_t}{\sqrt{\hat s_t}+\epsilon}
  $$

### 特点

* 收敛速度快
* 默认参数好用（$\beta _1=0.9,\ \beta _2=0.999,\ \eta=0.001$）
* 目前深度学习应用最广泛的优化器

---

## 11. 各方法总结与比较

| 算法                 | 特点                 | 优缺点           |
| ------------------ | ------------------ | ------------- |
| **GD**             | 使用全量梯度             | 精确但计算量大       |
| **SGD**            | 单样本更新              | 速度快但噪声大       |
| **Mini-batch SGD** | 折中方案               | GPU 高效并行      |
| **Momentum**       | 累积历史梯度             | 减少震荡，加快收敛     |
| **AdaGrad**        | 自适应学习率             | 稀疏特征好，学习率衰减过快 |
| **RMSProp**        | 平滑 AdaGrad         | 更适合非凸问题       |
| **Adadelta**       | 无需显式学习率            | 稳定，完全自适应      |
| **Adam**           | Momentum + RMSProp | 收敛快，最常用       |

---


