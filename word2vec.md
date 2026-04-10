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

### 方案 A：标准 Softmax（理论版，实际不用）

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

 one-hot          词嵌入向量             整个词表 V 个词
 编码的中心词      (即查表结果)           的概率分布
```

- 复杂度 $O(V)$：每次要算全部 $V$ 个词的得分 + softmax 分母

### 方案 B：Negative Sampling（实际常用）

```
输入层              隐藏层              输出：k+1 个 sigmoid
(V × 1)            (N × 1)

 ┌───┐              ┌───┐           W'中取 k+1 行
 │0  │              │   │         ┌──────────────────────┐
 │0  │     W        │   │    正样本│ σ(w'_pos · h) → 0.92 │ ← 期望→1
 │...│  ───────►    │ h │─────────│ σ(w'_neg₁· h) → 0.13 │ ← 期望→0
 │1  │  (N × V)     │   │    负样本│ σ(w'_neg₂· h) → 0.07 │ ← 期望→0
 │0  │              │   │         │ ...                   │
 │...│              └───┘         │ σ(w'_negₖ· h) → 0.05 │ ← 期望→0
 └───┘                            └──────────────────────┘

 one-hot          词嵌入向量        只看 1+k 个词，不看全部 V 个
```

- 保留 $W'$（$V \times N$），但每次只取其中 $k+1$ 行
- 复杂度 $O(k)$，$k$ 通常 5~20

### 方案 C：Hierarchical Softmax（论文原始方案）

```
输入层              隐藏层              输出：沿 Huffman 树走 log₂V 步
(V × 1)            (N × 1)

 ┌───┐              ┌───┐                  根: v_root
 │0  │              │   │                 ╱          ╲
 │0  │     W        │   │          σ(v_root·h)   1-σ(v_root·h)
 │...│  ───────►    │ h │──►      ╱                    ╲
 │1  │  (N × V)     │   │     "the"              节点C: v_C
 │0  │              │   │                       ╱          ╲
 │...│              └───┘                σ(v_C·h)      1-σ(v_C·h)
 └───┘                                 ╱                    ╲
                                   "cat"              节点B: v_B
 one-hot          词嵌入向量                         ╱          ╲
 编码的中心词      (即查表结果)                σ(v_B·h)      1-σ(v_B·h)
                                             ╱                    ╲
                                         "sat"              节点A: v_A
                                                           ╱          ╲
                                                       "on"          "mat"

 没有 W' 矩阵！参数是树的内部节点向量 v_root, v_C, v_B, v_A
```

- **去掉 $W'$**，改为 $(V-1)$ 个内部节点，每个节点一个 $N$ 维参数向量
- 预测一个词 = 沿路径连乘 sigmoid，例如：

$$P(\text{"sat"}) = (1-\sigma(\mathbf{v}_{root}^T \mathbf{h})) \times (1-\sigma(\mathbf{v}_C^T \mathbf{h})) \times \sigma(\mathbf{v}_B^T \mathbf{h})$$

- 每一步左+右=1，**整棵树自动归一化，没有分母**
- 复杂度 $O(\log V)$
- 高频词（如 "the"）离根近→路径短→算得快；低频词离根远但出现少

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

直接对整个词表做 softmax 计算量太大（$O(V)$），实际使用以下优化（两者都是**替换掉上面的 $W'$ 和输出层**，而非在其基础上加速）：

### 1. Negative Sampling（负采样）

- 不计算完整 softmax，而是把问题转化为**二分类**
- 对每个正样本（中心词-上下文词对），随机采 $k$ 个**负样本**（非上下文词）
- 仍然保留 $W'$（$V \times N$），但每次只查其中 $k+1$ 行，不产生完整的 $(V \times 1)$ 输出
- 损失函数变为：

$$\log \sigma(\mathbf{w}'^T_{pos} \cdot \mathbf{h}) + \sum_{i=1}^{k} \log \sigma(-\mathbf{w}'^T_{neg_i} \cdot \mathbf{h})$$

- $k$ 通常取 5~20（小语料取大值，大语料取小值）
- 复杂度：$O(k)$

### 2. Hierarchical Softmax（层次 softmax）

- **完全去掉 $W'$ 矩阵和 $(V \times 1)$ 输出层**，改用一棵二叉树
- 用**哈夫曼树**组织词表，每个**叶子**是一个词，每个**内部节点**有一个可学习的参数向量 $\mathbf{v}_n$（$N \times 1$）
- 预测一个词的概率 = 从根到该叶子的路径上，每个节点做一次 sigmoid 二分类，连乘起来
- 每个节点左+右概率 = 1，**自动归一化，不需要分母**
- 复杂度：$O(\log V)$

### 三种输出方式对比

| | 标准 Softmax | Negative Sampling | Hierarchical Softmax |
|---|---|---|---|
| 输出层参数 | $W'$：$V \times N$ | $W'$：$V \times N$（只用 $k$+1 行） | $(V-1)$ 个 $\mathbf{v}_n$：各 $N$ 维 |
| 输出向量 | $(V \times 1)$ 完整概率 | 无，只有 $k$+1 个 sigmoid | 无，只有路径上的 sigmoid 乘积 |
| 计算量/样本 | $O(V)$ | $O(k)$ | $O(\log V)$ |

---

## 一个具体例子

假设词表大小 $V = 10000$，嵌入维度 $N = 300$。

### 标准 Softmax 版

```
W  的维度: 300 × 10000  → 3,000,000 个参数
W' 的维度: 10000 × 300  → 3,000,000 个参数
总参数:                    6,000,000 个参数
```

输入 "cat" 的 one-hot 列向量 $\mathbf{x}$（第 42 个位置为 1）：
1. $\mathbf{h} = W \mathbf{x} = W$ 的第 42 列 → 得到 $(300 \times 1)$ 列向量
2. $\mathbf{u} = W' \mathbf{h}$ → 得到 $(10000 \times 1)$ 得分列向量
3. $\text{softmax}(\mathbf{u})$ → 10000 个概率值
4. 计算交叉熵损失，反向传播更新 $W$ 全部列 + $W'$ 全部行

### Negative Sampling 版

```
W  的维度: 300 × 10000    → 3,000,000 个参数
W' 的维度: 10000 × 300    → 3,000,000 个参数（但每次只用 k+1 行）
```

输入 "cat"，正样本 "sat"，随机采 $k=5$ 个负样本：
1. $\mathbf{h} = W$ 的第 42 列 → $(300 \times 1)$
2. 正样本得分：$\sigma(\mathbf{w}'_{sat}^T \mathbf{h})$ → 1 次点积
3. 负样本得分：$\sigma(\mathbf{w}'^T_{neg_i} \mathbf{h})$ → 5 次点积
4. 总共 **6 次**点积 + sigmoid，反向传播只更新 $W$ 的第 42 列 + $W'$ 的 6 行

### Hierarchical Softmax 版

```
W  的维度: 300 × 10000              → 3,000,000 个参数
树内部节点: 9999 个 × 300 维向量     → 2,999,700 个参数（没有 W'）
```

输入 "cat"，目标 "sat"，"sat" 在树中路径：根→右→右→左（3 步）
1. $\mathbf{h} = W$ 的第 42 列 → $(300 \times 1)$
2. 沿路径算 3 次 sigmoid 并连乘 → $P(\text{"sat"})$
3. 总共 **3 次**点积 + sigmoid，反向传播只更新 $W$ 的第 42 列 + 路径上 3 个 $\mathbf{v}_n$

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
