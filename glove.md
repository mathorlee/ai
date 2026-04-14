# GloVe（Global Vectors for Word Representation）

> 论文: *GloVe: Global Vectors for Word Representation* (Pennington, Socher, Manning, 2014, EMNLP)

## 核心思想

GloVe 的核心观点：**词向量应该编码的不是共现次数本身，而是共现概率的比值（ratio）**。

Word2Vec 是"局部"方法——每次只看一个窗口。GloVe 认为**全局共现统计**（整个语料的共现矩阵）包含了丰富的语义信息，应该直接利用。

GloVe = **基于矩阵分解的全局方法** + **类似 Word2Vec 的局部上下文思想** 的结合。

---

## 与 Word2Vec 的关键区别

| | Word2Vec | GloVe |
|---|---|---|
| 训练信号 | 局部上下文窗口（逐词扫描） | 全局共现矩阵（预先统计） |
| 优化目标 | 预测概率（softmax / sigmoid） | 加权最小二乘回归 |
| 本质 | 判别模型（预测任务） | 矩阵分解（回归任务） |
| 扫描方式 | 在线，逐窗口 | 离线，先统计再优化 |
| 训练速度 | 需要多次遍历语料 | 只需遍历非零共现条目 |

---

## 第一步：构建共现矩阵 $X$

### 定义

- $X_{ij}$ = 词 $j$ 出现在词 $i$ 的上下文窗口中的**总次数**（遍历整个语料统计）
- $X_i = \sum_k X_{ik}$ = 词 $i$ 上下文中所有词的总出现次数
- $P_{ij} = P(j|i) = X_{ij} / X_i$ = 词 $j$ 出现在词 $i$ 上下文中的**概率**

### 示例

假设语料为 `"the cat sat on the mat the cat sat"`, 窗口大小 = 1：

```
         the  cat  sat  on  mat
the   [   0    2    1    0    1  ]
cat   [   2    0    2    0    0  ]
sat   [   1    2    0    1    0  ]
on    [   0    0    1    0    1  ]
mat   [   1    0    0    1    0  ]
```

注意：
- $X$ 是**对称的**（$X_{ij} = X_{ji}$），因为 $i$ 出现在 $j$ 的上下文 ⟺ $j$ 出现在 $i$ 的上下文
- 实际中窗口内可以加**距离衰减**：距离为 $d$ 的共现贡献 $1/d$（GloVe 论文默认做法）

---

## 第二步：共现概率比值的语义洞察（论文核心直觉）

这是 GloVe 论文最精彩的部分。

### 例子：ice 与 steam

考虑两个目标词 $i = \text{ice}$, $j = \text{steam}$，观察不同探测词 $k$ 的**共现概率比值**：

| 探测词 $k$ | $P(k \mid \text{ice})$ | $P(k \mid \text{steam})$ | $P(k \mid \text{ice}) / P(k \mid \text{steam})$ |
|---|---|---|---|
| solid | $1.9 \times 10^{-4}$ | $2.2 \times 10^{-5}$ | **8.9** |
| gas | $6.6 \times 10^{-5}$ | $7.8 \times 10^{-4}$ | **8.5 × 10⁻²** |
| water | $3.0 \times 10^{-3}$ | $2.2 \times 10^{-3}$ | **1.36** |
| fashion | $1.7 \times 10^{-5}$ | $1.8 \times 10^{-5}$ | **0.96** |

**解读**：
- $k = \text{solid}$：和 ice 相关、和 steam 不相关 → 比值**远大于 1**
- $k = \text{gas}$：和 steam 相关、和 ice 不相关 → 比值**远小于 1**
- $k = \text{water}$：和两者都相关 → 比值**接近 1**
- $k = \text{fashion}$：和两者都不相关 → 比值也**接近 1**

**结论**：单独看 $P(k|i)$ 或 $P(k|j)$ 噪声大，但**比值** $P(k|i)/P(k|j)$ 能清晰区分语义关系。

---

## 第三步：从比值到模型（论文推导）

### 3.1 定义目标函数的形式

我们希望找一个函数 $F$，使得词向量能编码比值信息：

$$F(\mathbf{w}_i, \mathbf{w}_j, \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}$$

其中 $\mathbf{w}_i, \mathbf{w}_j$ 是目标词向量，$\tilde{\mathbf{w}}_k$ 是上下文词向量。

### 3.2 要求 $F$ 在向量差上操作

因为比值编码的是 $i$ 和 $j$ 的**差异**，所以 $F$ 应该依赖于 $(\mathbf{w}_i - \mathbf{w}_j)$：

$$F(\mathbf{w}_i - \mathbf{w}_j, \tilde{\mathbf{w}}_k) = \frac{P_{ik}}{P_{jk}}$$

### 3.3 将向量映射为标量

左边参数是**向量**，右边是**标量**。最自然的方式：做**点积**：

$$F\big((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k\big) = \frac{P_{ik}}{P_{jk}}$$

### 3.4 要求 $F$ 是指数函数

注意右边是**比值**（除法），左边是**差**（减法）。为了让减法映射到除法，$F$ 需要满足**同态**性质：

$$F(a - b) = \frac{F(a)}{F(b)}$$

满足这个性质的函数是 $F = \exp$：

$$\exp\big((\mathbf{w}_i - \mathbf{w}_j)^T \tilde{\mathbf{w}}_k\big) = \frac{P_{ik}}{P_{jk}}$$

展开：

$$\exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_k - \mathbf{w}_j^T \tilde{\mathbf{w}}_k) = \frac{\exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_k)}{\exp(\mathbf{w}_j^T \tilde{\mathbf{w}}_k)} = \frac{P_{ik}}{P_{jk}}$$

所以只需满足：

$$\exp(\mathbf{w}_i^T \tilde{\mathbf{w}}_k) = P_{ik} = \frac{X_{ik}}{X_i}$$

取对数：

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_k = \log P_{ik} = \log X_{ik} - \log X_i$$

### 3.5 吸收常数项

$\log X_i$ 只依赖于 $i$，不依赖于 $k$，可以吸收为偏置项 $b_i$。为了恢复对称性（因为 $X$ 是对称的），也为 $k$ 加一个偏置 $\tilde{b}_k$：

$$\boxed{\mathbf{w}_i^T \tilde{\mathbf{w}}_k + b_i + \tilde{b}_k = \log X_{ik}}$$

**这就是 GloVe 的核心等式**：词向量的点积 + 偏置 = 共现次数的对数。

---

## 第四步：损失函数

直接用最小二乘：

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \big(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\big)^2$$

### 加权函数 $f(x)$

为什么需要 $f$？
- $X_{ij} = 0$ 时 $\log 0 = -\infty$，需要跳过（$f(0) = 0$）
- 高频共现（如 "the the"）不应主导损失
- 低频共现应有合理权重，不能忽略

论文使用的加权函数：

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

```
f(x)
 1 ┤·····························───────────────
   │                          ╱
   │                        ╱
   │                      ╱
   │                   ╱
   │                ╱
   │            ╱
   │        ╱
   │    ╱
 0 ┤╱─────────┬──────────┬──────────────────
   0        x_max                          x
```

论文推荐参数：$x_{\max} = 100$，$\alpha = 3/4$。

- $\alpha = 3/4$ 比 $\alpha = 1$（线性）效果好，因为它对中频词给了更高的相对权重

---

## 模型参数

| 参数 | 维度 | 含义 |
|---|---|---|
| $W$ | $V \times d$ | 目标词嵌入矩阵，第 $i$ 行为 $\mathbf{w}_i$ |
| $\tilde{W}$ | $V \times d$ | 上下文词嵌入矩阵，第 $j$ 行为 $\tilde{\mathbf{w}}_j$ |
| $\mathbf{b}$ | $V \times 1$ | 目标词偏置 |
| $\tilde{\mathbf{b}}$ | $V \times 1$ | 上下文词偏置 |
| **总参数量** | $2Vd + 2V$ | 比 Word2Vec 多了偏置项 |

训练完成后，论文建议使用 $\mathbf{w}_i + \tilde{\mathbf{w}}_i$ 作为最终词向量（因为 $X$ 对称，两个矩阵角色等价，求和可以减少噪声）。

---

## 训练流程

```
步骤 1: 遍历语料，统计共现矩阵 X（只需一次扫描）
         ↓
步骤 2: 提取所有非零条目 (i, j, X_ij)
         ↓
步骤 3: 随机初始化 W, W̃, b, b̃
         ↓
步骤 4: 对非零条目做 SGD / AdaGrad：
         ┌──────────────────────────────────────────────────┐
         │ for each (i, j, X_ij) in 非零条目:              │
         │   pred = w_i · w̃_j + b_i + b̃_j                │
         │   loss = f(X_ij) * (pred - log(X_ij))²          │
         │   更新 w_i, w̃_j, b_i, b̃_j（AdaGrad）          │
         └──────────────────────────────────────────────────┘
         ↓
步骤 5: 最终词向量 = w_i + w̃_i
```

注意：
- GloVe 论文使用 **AdaGrad** 优化器（自适应学习率）
- 只遍历**非零条目**，不需要遍历整个 $V \times V$ 矩阵
- 非零条目数远远小于 $V^2$（共现矩阵是稀疏的）

---

## 论文超参数与结果

### 训练配置
- 语料：6B tokens（Wikipedia 2014 + Gigaword 5）和更大的 42B tokens（Common Crawl）
- 词表大小：400K
- 窗口大小：10（左 10 右 10）
- 向量维度：50 / 100 / 200 / 300
- $x_{\max} = 100$，$\alpha = 3/4$
- 迭代：50 次或 100 次
- 学习率：0.05（AdaGrad）

### 主要结论
- 300d + 6B tokens 在词类比任务上达到 **75%** 准确率（Word2Vec Skip-gram 约 64%）
- 42B tokens 的 300d 模型在词类比任务上达到 **81.9%**
- 向量维度增加到超过 200 之后收益递减
- 窗口大小对**对称型**窗口在语法任务上更好；**非对称型**对语义任务略好
- 更多数据 > 更大维度

---

## 与 Word2Vec 的数学联系

GloVe 论文指出，Skip-gram with Negative Sampling（SGNS）实际上也在隐式地分解一个矩阵。

Levy & Goldberg (2014) 证明：SGNS 等价于分解一个**偏移的 PMI 矩阵**：

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_j = \text{PMI}(i,j) - \log k$$

其中 PMI（逐点互信息）为：

$$\text{PMI}(i,j) = \log \frac{P(i,j)}{P(i) \cdot P(j)} = \log \frac{X_{ij} \cdot |\text{corpus}|}{X_i \cdot X_j}$$

而 GloVe 分解的是：

$$\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j = \log X_{ij}$$

两者本质上都在对共现统计的对数做低秩分解，只是形式不同。

---

## GloVe 的优势与局限

### 优势
1. **训练高效**：只需遍历非零共现条目，训练时间与 $|X_{\text{nonzero}}|$ 成正比，而非语料大小
2. **利用全局信息**：共现矩阵整合了整个语料的统计信息
3. **可并行**：损失函数可以对条目并行计算
4. **可解释性强**：目标函数有清晰的数学推导（从比值出发）
5. **小语料也能表现好**：因为全局统计比逐窗口采样更稳定

### 局限
1. **内存开销**：需要存储共现矩阵（稀疏矩阵，但仍然随词表增大）
2. **静态嵌入**：和 Word2Vec 一样，每个词只有一个向量，无法处理多义词
3. **无法处理 OOV**：词表外的词没有向量（后来 FastText 用子词解决）
4. **预处理时间**：统计共现矩阵需要一次全量遍历

---

## 完整示例

假设词表大小 $V = 10000$，嵌入维度 $d = 300$。

### 参数量

```
W  的维度: 10000 × 300    → 3,000,000 个参数（目标词嵌入）
W̃ 的维度: 10000 × 300    → 3,000,000 个参数（上下文词嵌入）
b  的维度: 10000 × 1      →    10,000 个参数（目标词偏置）
b̃ 的维度: 10000 × 1      →    10,000 个参数（上下文词偏置）
总参数:                      6,020,000 个参数
```

### 训练一个条目

对于 $i = \text{cat}$（第 42 号），$j = \text{sat}$（第 107 号），$X_{42,107} = 15$：

```
1. 取出 w_42 (300×1), w̃_107 (300×1), b_42 (标量), b̃_107 (标量)
2. 计算预测值: pred = w_42 · w̃_107 + b_42 + b̃_107
3. 计算目标值: target = log(15) = 2.708
4. 计算权重:   f(15) = (15/100)^0.75 = 0.150^0.75 = 0.227
5. 计算损失:   loss = 0.227 × (pred - 2.708)²
6. 用 AdaGrad 更新 w_42, w̃_107, b_42, b̃_107
```

### 与 Word2Vec 对比

| | Word2Vec (Skip-gram + NS) | GloVe |
|---|---|---|
| 输入 | 逐窗口扫描：(cat, sat) 出现 1 次算 1 次 | 预先统计：$X_{\text{cat,sat}} = 15$，只算 1 次 |
| 每步计算 | 1 次正样本 + $k$ 次负样本的 sigmoid | 1 次点积 + 平方误差 |
| 目标 | 最大化 $\log \sigma(\mathbf{w}^T \tilde{\mathbf{w}})$ | 最小化 $(\mathbf{w}^T \tilde{\mathbf{w}} + b + \tilde{b} - \log X)^2$ |
| 训练遍历 | 语料中每个位置 | 共现矩阵中每个非零条目 |
