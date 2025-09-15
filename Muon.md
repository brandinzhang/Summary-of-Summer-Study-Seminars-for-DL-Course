
---

## 一、《Muon优化器赏析：从向量到矩阵的本质跨越》（2024-12-10）



### 1. Muon 的基础更新结构

* Muon 全称为 **MomentUm Orthogonalized by Newton–Schulz**，特别适用于处理矩阵参数 $\mathbf{W} \in \mathbb{R}^{n \times m}$。
* 更新规则形式如下：

  $$
  \mathbf{M}_t = \beta \mathbf{M}_{t-1} + \mathbf{G}_t
  $$

  $$
  \mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \left[ \mathrm{msign}(\mathbf{M}_t) + \lambda \mathbf{W}_{t-1} \right]
  $$

  其中，$\mathrm{msign}(\cdot)$ 是矩阵符号函数，而不是逐元素的 sign 操作，而是基于 SVD 的矩阵级处理方式 ([科学空间][1])。

### 2. 矩阵符号函数 msign 的定义与性质

* 利用 SVD 分解：若 $\mathbf{M} = \mathbf{U} \Sigma \mathbf{V}^\top$，则

  $$
  \mathrm{msign}(\mathbf{M}) = \mathbf{U}_{[:, :r]} \mathbf{V}_{[:, :r]}^\top
  $$

* 它可以等价表示为：

  $$
  \mathrm{msign}(\mathbf{M}) = (\mathbf{M} \mathbf{M}^\top)^{-1/2} \mathbf{M} = \mathbf{M}(\mathbf{M}^\top \mathbf{M})^{-1/2}
  $$

  这推广了标量 $\text{sign}(x) = x (x^2)^{-1/2}$ 的概念 ([科学空间][1])。

* 当 $\mathbf{M}$ 为对角矩阵时，msign 会退化成元素级的 sign 操作；当作为列向量（$n\times1$ 矩阵），则等同于 $L^2$ 归一化 ([科学空间][1])。

* 当矩阵为方阵时，msign 是将 $\mathbf{M}$ 最近似的正交矩阵，即极分解中的正交部分：

  $$
  \mathrm{msign}(\mathbf{M}) = \arg\min_{\mathbf{O}^\top \mathbf{O} = I} \|\mathbf{M} - \mathbf{O}\|_F^2 = \mathbf{U}\mathbf{V}^\top
  $$

  验证了它是最优正交近似 ([科学空间][1])。

### 3. 迭代近似：Newton–Schulz 迭代

* 直接计算 SVD 成本高，在实践中使用 Newton–Schulz 迭代来近似 msign：

  $$
  \mathbf{X}_{t+1} = \frac{15}{8} \mathbf{X}_t - \frac{5}{4} \mathbf{X}_t (\mathbf{X}_t^\top \mathbf{X}_t) + \frac{3}{8} \mathbf{X}_t (\mathbf{X}_t^\top \mathbf{X}_t)^2
  $$

  而 Muon 实际源码中用的系数是 $(3.4445,\,-4.7750,\ 2.0315)$，据苏剑林猜测，这大概是迭代步数为5时仿真的最优解。 ([科学空间][1])。

---

## 二、《Muon续集：为什么我们选择尝试Muon？》（2025-02-27）


### 1. “最小作用量原理”下的 Muon

* 理想优化器应在二维度上同时满足：

  1. **稳**：每步对模型扰动尽可能小；
  2. **快**：每步对 loss 降低尽可能大。

* 转化为以下受约束的优化问题：

  $$
  \arg\min_{\Delta W} \mathrm{Tr}(G^\top \Delta W) \quad \text{s.t. } \rho(\Delta W) \le \eta
  $$

  其中 $\Delta\boldsymbol{W}_{t+1}=\boldsymbol{W}_{t+1}-\boldsymbol{W}_t,\boldsymbol{G}_t=\nabla_{\boldsymbol{W}_t}\mathcal{L}(\boldsymbol{W}_t), \rho$ 是对更新稳健性的度量，$\eta$ 可视为学习率 ([科学空间][2])。

* 若以谱范数 $\| \Delta W \|_2$ 作为度量，则解为：

  $$
  \Delta W = -\eta\, \mathrm{msign}(G) = -\eta\, U V^\top
  $$

  如加入动量 $\beta$，则用 $\mathrm{msign}(M)$，得出“Muon 是谱范数下的最速下降” ([科学空间][2])。

### 2. 权重衰减（Weight Decay）

* 原始 Muon 未加衰减，训练前期表现优，但容易被 Adam 追上，甚至出现训练不稳定。
* 添加权重衰减后的更新规则：

  $$
  \Delta W = -\eta\, [\mathrm{msign}(M) + \lambda W]
  $$

  有助于控制参数谱范数，使 $\|W\|_2$ 保持有界，保障训练稳定性，特别对防止 Attention logits 爆炸尤为重要 ([科学空间][2])。

### 3. RMS 对齐（Update RMS Alignment）

* 为方便迁移 Adam 的超参数，提出用 RMS 对齐方式调整 Muon 更新幅度：

  * 定义矩阵 RMS：

    $$
    \mathrm{RMS}(W) = \frac{\|W\|_F}{\sqrt{nm}}
    $$
  * 将 Muon 更新量对齐到 Adam 的 RMS 范围（约 0.2～0.4），推荐形式：

    $$
    W_t = W_{t-1} - \eta_t (0.2\, O_t\, \sqrt{\max(n,m)} + \lambda W_{t-1})
    $$
  * 尤其在 MoE 模型中，不同维度参数适合使用不同尺度的学习率调整 ([科学空间][3])。

### 4. 实验分析与观察

* 在 2.4B / 16B MoE 上，Muon 在训练收敛速度与性能上明显优于 Adam ([科学空间][2])。
* Muon 训练出的参数奇异值分布更为均匀（用奇异值熵衡量），说明参数空间更充实，减少“塌缩”倾向 ([科学空间][2])。
* 在预训练与微调组合上，使用 Muon 预训练且 Muon 微调效果最佳；其他组合（如 Adam + Muon）效果不稳定，表明初始化与优化器匹配存在一定挑战 ([科学空间][2])。

### 5. 后续思考与拓展

* Adam 与模型架构可能渐进“共同演化”，Muon 的成功提供了另一路径值得关注 ([科学空间][2])。
* 后续可探索：

  * 使用其他范数（如 Schatten-norm）替代谱范数；
  * 调整 µP（maximal update parametrization）参数设计；
  * 更深入理解优化器稳定性与初始化匹配机制 ([科学空间][2])。

---

[1]: https://spaces.ac.cn/archives/10592 "Muon优化器赏析：从向量到矩阵的本质跨越 - 科学空间"
[2]: https://spaces.ac.cn/archives/10739 "Muon续集：为什么我们选择尝试Muon？ - 科学空间|Scientific Spaces"
[3]: https://spaces.ac.cn/archives/10739/comment-page-1 "Muon续集：为什么我们选择尝试Muon？ - 科学空间|Scientific Spaces"
