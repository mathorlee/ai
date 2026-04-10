# Word2Vec

> 论文: *Efficient Estimation of Word Representations in Vector Space* (Mikolov et al., 2013, arXiv:1301.3781v3)

## 核心思想

Word2Vec 是一种将词语映射为稠密向量（词嵌入）的方法。核心假设：**语义相近的词，上下文也相似**。

---

## 论文关键要点

### 动机
- 传统 NLP 把词当作原子单元（one-hot），无法表达词间相似性
- 已有神经网络语言模型（NNLM、RNNLM）计算复杂度太高，瓶颈在非线性隐藏层 $H \times V$
- 目标：去掉隐藏层，用更简单的 log-linear 模型在更大数据上更快训练

### 计算复杂度对比（论文 Section 2-3）

| 模型 | 训练复杂度 $Q$ |
|---|---|
| NNLM | $N \times D + N \times D \times H + H \times V$ |
| RNNLM | $H \times H + H \times V$ |
| **CBOW** | $N \times D + D \times \log_2(V)$ |
| **Skip-gram** | $C \times (D + D \times \log_2(V))$ |

总训练复杂度 $O = E \times T \times Q$，其中 $E$ 为 epoch 数，$T$ 为训练词数。

### 论文实验超参数
- 训练数据: Google News 语料 ~6B tokens
- 词表: 限制到最频繁的 1M 词
- 嵌入维度: 300 / 600 / 1000
- 窗口大小: CBOW 用 4（前4后4），Skip-gram 用 $C=5$ 或 $C=10$（随机取 $R \in [1, C]$）
- 学习率: 0.025，**线性衰减到 0**
- Epochs: 3（大数据集用 1 epoch 效果相当）
- 使用 Hierarchical Softmax（Huffman 树）

### 论文核心结果（Table 4/5）
- Skip-gram 300d + 783M words: 语义准确率 50.0%，语法准确率 55.9%
- CBOW 训练更快（约1天），Skip-gram 更慢（约3天）
- Skip-gram 在语义任务上明显优于 CBOW
- 1 epoch 在 1.6B 词上训练 ≈ 3 epochs 在 783M 词上训练

---

## 两种架构

### 1. CBOW（Continuous Bag of Words）

用 **上下文词** 预测 **中心词**。

```
输入: [the, cat, on, mat]  →  预测: sat
```

### 2. Skip-gram

用 **中心词** 预测 **上下文词**。

```
输入: sat  →  预测: [the, cat, on, mat]
```

---

## 网络结构（以 Skip-gram 为例）

```
输入层              隐藏层(投影层)           输出层
(V × 1)            (N × 1)               (V × 1)

 ┌───┐              ┌───┐                 ┌───┐
 │0  │              │   │                 │p₁ │
 │0  │     W        │   │      W'         │p₂ │
 │...│  ───────►    │ h │  ───────►       │...│
 │1  │  (N × V)     │   │  (V × N)        │pⱼ │  ← softmax
 │0  │              │   │                 │...│
 │...│              └───┘                 │pᵥ │
 └───┘                                    └───┘

 one-hot          词嵌入向量             上下文词的
 编码的中心词      (即查表结果)           概率分布
```

| 层 | 维度 | 说明 |
|---|---|---|
| 输入层 | $V \times 1$ | 中心词的 one-hot 向量，$V$ = 词表大小 |
| 隐藏层 | $N \times 1$ | 嵌入向量，$N$ = 嵌入维度（超参数，常取 100~300） |
| 输出层 | $V \times 1$ | 经过 softmax 后的概率分布 |

---

## 网络参数 / 权重矩阵

整个网络**只有两个权重矩阵**，没有偏置，没有激活函数（隐藏层是线性的）：

### 权重矩阵 $W$（输入→隐藏）

- **维度**：$N \times V$
- **角色**：**输入嵌入矩阵**（也叫 embedding matrix）
- 矩阵的**第 $i$ 列** $\mathbf{w}_i$ 就是词表中第 $i$ 个词的嵌入向量（$N \times 1$ 列向量）
- 对于 one-hot 输入 $\mathbf{x}$（$V \times 1$），隐藏层输出为：

$$\mathbf{h} = W \mathbf{x} = \mathbf{w}_i \quad (N \times V)(V \times 1) = (N \times 1)$$

这本质上就是**查表（lookup）**操作，取出对应列。

### 权重矩阵 $W'$（隐藏→输出）

- **维度**：$V \times N$
- **角色**：**输出嵌入矩阵**（也叫 context matrix）
- 矩阵的**第 $j$ 行** $\mathbf{w}'^T_j$ 是词表中第 $j$ 个词作为上下文时的向量表示（$\mathbf{w}'_j$ 本身是 $N \times 1$ 列向量）
- 输出层的得分向量：

$$\mathbf{u} = W' \mathbf{h} \quad (V \times N)(N \times 1) = (V \times 1)$$

- 其中第 $j$ 个元素为两个列向量的点积：

$$u_j = \mathbf{w}'^T_j \mathbf{h}$$

- 经过 softmax 得到概率：

$$p(w_j | w_i) = \frac{\exp(u_j)}{\sum_{k=1}^{V} \exp(u_k)}$$

### 参数总结

| 参数 | 维度 | 含义 |
|---|---|---|
| $W$ | $N \times V$ | 每一**列**是一个词的**输入嵌入**向量（$N \times 1$） |
| $W'$ | $V \times N$ | 每一**行**是一个词的**输出嵌入**向量（$1 \times N$） |
| **总参数量** | $2 \times V \times N$ | 两个矩阵，无偏置 |

> 训练完成后，通常**只取 $W$ 的列**作为最终的词向量。也有做法将 $W$ 和 $W'^T$ 取平均。

---

## 训练目标

最大化在给定中心词情况下，正确上下文词出现的概率：

$$J = -\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \le j \le c, j \ne 0} \log p(w_{t+j} | w_t)$$

其中 $T$ 为语料中的总词数，$c$ 为窗口大小。

---

## 训练加速技巧

直接对整个词表做 softmax 计算量太大（$O(V)$），实际使用以下优化：

### 1. Negative Sampling（负采样）

- 不计算完整 softmax，而是把问题转化为**二分类**
- 对每个正样本（中心词-上下文词对），随机采 $k$ 个**负样本**（非上下文词）
- 损失函数变为：

$$\log \sigma(\mathbf{w}'^T_{pos} \cdot \mathbf{h}) + \sum_{i=1}^{k} \log \sigma(-\mathbf{w}'^T_{neg_i} \cdot \mathbf{h})$$

- $k$ 通常取 5~20（小语料取大值，大语料取小值）

### 2. Hierarchical Softmax（层次 softmax）

- 用**哈夫曼树**组织词表，将 softmax 从 $O(V)$ 降到 $O(\log V)$
- 每个叶子节点是一个词，从根到叶子的路径上做一系列二分类

---

## 一个具体例子

假设词表大小 $V = 10000$，嵌入维度 $N = 300$：

```
W  的维度: 300 × 10000  → 3,000,000 个参数
W' 的维度: 10000 × 300  → 3,000,000 个参数
总参数:                    6,000,000 个参数
```

输入 "cat" 的 one-hot 列向量 $\mathbf{x}$（第 42 个位置为 1）：
1. $\mathbf{h} = W \mathbf{x} = W$ 的第 42 列 → 得到 $(300 \times 1)$ 列向量
2. $\mathbf{u} = W' \mathbf{h}$ → 得到 $(10000 \times 1)$ 得分列向量
3. $\text{softmax}(\mathbf{u})$ → 10000 个概率值（仍是 $V \times 1$ 列向量）
4. 与真实上下文词计算交叉熵损失，反向传播更新 $W$ 和 $W'$

---

## CBOW vs Skip-gram 对比

| | CBOW | Skip-gram |
|---|---|---|
| 输入 | 多个上下文词 | 单个中心词 |
| 输出 | 预测中心词 | 预测多个上下文词 |
| 隐藏层 | 上下文词向量的**平均** | 中心词向量直接作为 $\mathbf{h}$ |
| 训练速度 | 更快 | 更慢 |
| 低频词效果 | 较差 | **更好** |
| 适用场景 | 大语料 | 小语料 / 需要捕捉罕见词 |
