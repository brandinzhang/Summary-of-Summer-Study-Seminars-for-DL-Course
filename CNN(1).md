- 数学建模B题：https://spaces.ac.cn/archives/4100


# 卷积的实质：学习频域上的模式


## 回顾卷积
在经典信号处理理论中，卷积定义为系统对历史输入的累积响应。给定输入信号 $f(t)$ 与核函数 $g(t)$，卷积运算严格表述为：

$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t - \tau)  d\tau$

其核心在于核函数的翻转操作（ $t-\tau$ 项体现时间反演）。例如在声学系统中，该操作表征声波衰减的历史累积效应。

举个例子：一个人任一瞬时的进食数量如下：
![20250723155333](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250723155333.png)

食物消化的速率如下图，此图事实上展示的是衰减速率：

![20250723155708](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250723155708.png)

我们如何求某一时刻$t$人肚子里剩余的食物量？以下午十四点为例子，想要得到这一时刻剩余食物量，应当考虑的是此前所有时刻的进食影响和衰减影响。![20250723155830](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250723155830.png)

这就是卷积的本质。而在卷积神经网络(CNN)中，实际执行的运算本质是互相关：

$S(i,j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} I(i+m, j+n) \cdot K(m,n)$

其中  $I$  为输入图像， $K$  为卷积核。这里省去核翻转的物理意义源于两个关键特性：
1. 权重矩阵  $K$  通过梯度下降自动学习最优方向
2. 数学上可证明翻转操作不影响特征检测能力（Zeiler & Fergus, 2014）

## 图像的频率

图像作为二维离散信号，其频域表征通过二维离散傅里叶变换实现：

$$F(u,v) = \frac{1}{\sqrt{MN}} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) e^{-i2\pi (\frac{ux}{M} + \frac{vy}{N})}$$

其中关键物理概念为：
• 低频分量：对应空间梯度小的区域 $\nabla I \approx 0$

   $$\text{示例：蓝天区域} \quad \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \to 0$$
• 高频分量：对应拉普拉斯算子响应强的区域 

  $$\text{边缘满足} \quad \nabla I > \gamma \quad \text{(e.g. Sobel算子响应)}$$
• 频率分布：自然图像的功率谱呈现1/f^2分布，低频成分占主导

   $$\text{功率谱} \quad P(f) \propto \frac{1}{f^2} \quad f\text{为空间频率}$$
这表明低频分量包含了图像的主要结构信息，而高频分量则包含细节和噪声。
实验测得自然图像的功率谱呈 1/f^2 分布（Field, 1987）：

$$P(f) \propto \frac{1}{f^2} \quad f\text{为空间频率}$$

这解释了为什么低频分量占总能量的70%以上。

## 卷积核的模式检测原理

考虑3×3卷积核对输入块 $\mathbf{X} \in \mathbb{R}^{3\times3}$ 的运算：

$$\text{Response} = \langle \mathbf{X}, \mathbf{K} \rangle_F = \sum_{i=1}^3 \sum_{j=1}^3 X_{ij}K_{ij}$$

该点积运算本质是余弦相似度的缩放形式：

$$\text{Sim}(\mathbf{X},\mathbf{K}) = \frac{\langle \mathbf{X}, \mathbf{K} \rangle}{\|\mathbf{X}\| \|\mathbf{K}\|}$$

当固定核权重时，响应峰值对应与核模式最相似的图像块。

层级抽象过程可表述为：

$$
\begin{align*}
\text{浅层特征} \, \mathcal{F}_1 &= \sigma(\mathbf{W}_1 \ast \mathbf{I}) \quad \text{(边缘/纹理)} \\
\text{深层特征} \, \mathcal{F}_L &= \Phi(\mathbf{W}_L \ast \mathcal{F}_{L-1}) \quad \text{(语义部件组合)}
\end{align*}
$$
其中 $\Phi(\cdot)$ 表示非线性组合函数。Zeiler的可视化实验证明：当输入为 $\mathbf{I} = \mathbf{K}_l \text{(第l层核)}$，第 l 层特征图的峰值信噪比(PSNR)可达35dB以上，验证了层级特征匹配机制。



神经正切核(NTK)理论（Jacot et al., 2018）揭示了收敛速率的频域差异。设损失函数在频域的分解形式：

$$
\mathcal{L}(\theta) = \sum_k \langle \psi_k, \theta \rangle^2 \quad k\in\text{频段索引}
$$
其中特征函数 $\psi_k$ 满足：
$$
\nabla_\theta \mathcal{L} \cdot \psi_k = \lambda_k \psi_k
$$
则梯度下降的动态方程为：
$$
\frac{d\theta_t}{dt} = -\nabla \mathcal{L}(\theta_t) \Rightarrow \frac{d\langle \psi_k,\theta_t\rangle}{dt} = -\lambda_k \langle \psi_k,\theta_t\rangle
$$
解为指数衰减：
$$
\langle \psi_k,\theta_t\rangle = \langle \psi_k,\theta_0\rangle e^{-\lambda_k t}
$$
其中 $\lambda_k$ 为频段 k 的收敛速率。由于自然图像的低频分量 $\lambda_{\text{low}} > \lambda_{\text{high}}$，低频分量以 $e^{-\lambda_{\text{low}} t}$ 快速收敛，而高频分量以 $e^{-\lambda_{\text{high}} t}$ 缓慢收敛。临界结论：由于自然图像的 $\lambda_{\text{低频}} > \lambda_{\text{高频}}$，低频分量以  $e^{-\lambda_{\text{low}} t}$  快速收敛，而高频分量以  $e^{-\lambda_{\text{high}} t}$  缓慢收敛。


推荐论文：《Theory of the frequency principle for general deep neural networks》

