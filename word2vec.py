"""
Word2Vec PyTorch 实现
基于论文: Efficient Estimation of Word Representations in Vector Space (Mikolov et al., 2013)
实现了 Skip-gram 和 CBOW 两种架构，支持 Negative Sampling 和简易训练流程。

训练数据: Text8 (Wikipedia 前 100MB 清洗文本，约 17M 词)
下载地址: http://mattmahoney.net/dc/text8.zip
"""

import os
import math
import zipfile
import urllib.request
import collections
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. 数据下载与预处理
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"
TEXT8_PATH = os.path.join(DATA_DIR, "text8.zip")


def download_text8():
    """下载 Text8 数据集"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TEXT8_PATH):
        print(f"正在下载 Text8 数据集到 {TEXT8_PATH} ...")
        urllib.request.urlretrieve(TEXT8_URL, TEXT8_PATH)
        print("下载完成。")
    else:
        print("Text8 数据集已存在，跳过下载。")


def read_text8() -> List[str]:
    """读取 Text8，返回词列表"""
    download_text8()
    with zipfile.ZipFile(TEXT8_PATH) as f:
        text = f.read(f.namelist()[0]).decode("utf-8")
    words = text.strip().split()
    print(f"语料总词数: {len(words):,}")
    return words


class Vocabulary:
    """
    构建词表，支持:
    - min_count 过滤低频词
    - subsampling 高频词下采样 (论文公式: P(discard) = 1 - sqrt(t / f(w)))
    - 负采样的 unigram 分布 (频率的 3/4 次方)
    """

    def __init__(self, words: List[str], min_count: int = 5, max_vocab_size: int = 50000):
        counter = collections.Counter(words)
        # 按频率降序，截取 max_vocab_size
        sorted_vocab = sorted(counter.items(), key=lambda x: -x[1])
        sorted_vocab = [(w, c) for w, c in sorted_vocab if c >= min_count]
        if len(sorted_vocab) > max_vocab_size:
            sorted_vocab = sorted_vocab[:max_vocab_size]

        self.word2idx = {}
        self.idx2word = []
        self.word_counts = []

        for idx, (word, count) in enumerate(sorted_vocab):
            self.word2idx[word] = idx
            self.idx2word.append(word)
            self.word_counts.append(count)

        self.word_counts = np.array(self.word_counts, dtype=np.float64)
        self.total_words = self.word_counts.sum()
        self.vocab_size = len(self.idx2word)

        # 构建负采样 unigram 分布: freq^0.75 (论文 [21] 中提出)
        freqs = self.word_counts ** 0.75
        self.neg_sampling_probs = freqs / freqs.sum()

        # Subsampling 概率: 高频词以一定概率被丢弃
        # P(keep) = sqrt(t / f(w)) + t / f(w), 其中 t 是阈值 (论文用 1e-5)
        t = 1e-5
        word_freqs = self.word_counts / self.total_words
        self.keep_probs = np.sqrt(t / word_freqs) + (t / word_freqs)
        self.keep_probs = np.minimum(self.keep_probs, 1.0)

        print(f"词表大小: {self.vocab_size:,} (min_count={min_count})")

    def encode(self, words: List[str]) -> List[int]:
        """将词列表转为索引列表，忽略 OOV 词"""
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def subsample(self, word_indices: List[int]) -> List[int]:
        """对高频词进行下采样"""
        return [idx for idx in word_indices if random.random() < self.keep_probs[idx]]

    def negative_sample(self, num_samples: int) -> np.ndarray:
        """按 unigram^(3/4) 分布采样负样本"""
        return np.random.choice(self.vocab_size, size=num_samples, p=self.neg_sampling_probs)


# ============================================================
# 2. 数据集
# ============================================================

class SkipGramDataset(Dataset):
    """
    Skip-gram 数据集:
    对每个中心词，在 [1, window_size] 范围内随机选 R，
    取前后各 R 个词作为正样本对。
    """

    def __init__(self, word_indices: List[int], window_size: int = 5):
        self.data = []
        print("正在构建 Skip-gram 训练对...")
        for i in range(len(word_indices)):
            center = word_indices[i]
            # 论文: 随机选 R in [1, C]，用前后各 R 个词
            R = random.randint(1, window_size)
            for j in range(max(0, i - R), min(len(word_indices), i + R + 1)):
                if j != i:
                    self.data.append((center, word_indices[j]))
        print(f"训练样本数: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        center, context = self.data[idx]
        return center, context


class CBOWDataset(Dataset):
    """
    CBOW 数据集:
    用上下文窗口内的词（前后各 window_size 个）预测中心词。
    论文中使用 window_size=4（前4后4）。
    """

    def __init__(self, word_indices: List[int], window_size: int = 4):
        self.data = []
        self.window_size = window_size
        print("正在构建 CBOW 训练对...")
        for i in range(window_size, len(word_indices) - window_size):
            context = []
            for j in range(i - window_size, i + window_size + 1):
                if j != i:
                    context.append(word_indices[j])
            target = word_indices[i]
            self.data.append((context, target))
        print(f"训练样本数: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), target


# ============================================================
# 3. 模型定义
# ============================================================

class SkipGramNegSampling(nn.Module):
    """
    Skip-gram with Negative Sampling

    论文中的两个权重矩阵:
    - W  (input embeddings):  vocab_size × embed_dim  → 即 self.in_embeddings
    - W' (output embeddings): vocab_size × embed_dim  → 即 self.out_embeddings

    前向过程:
    1. 从 W 中查表得到中心词向量 h = W[center]
    2. 对正样本: score_pos = dot(W'[pos], h) → sigmoid → 期望接近 1
    3. 对负样本: score_neg = dot(W'[neg], h) → sigmoid → 期望接近 0

    损失函数 (Negative Sampling Loss):
      L = -log σ(W'[pos]·h) - Σ log σ(-W'[neg_i]·h)
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # W: 输入嵌入矩阵 (论文的 embedding lookup)
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        # W': 输出嵌入矩阵 (论文的 context matrix)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)

        # 初始化: 均匀分布 [-0.5/dim, 0.5/dim]
        init_range = 0.5 / embed_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.zero_()

    def forward(self, center: torch.Tensor, pos_context: torch.Tensor,
                neg_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            center:      (batch_size,)          中心词索引
            pos_context: (batch_size,)          正样本上下文词索引
            neg_context: (batch_size, num_neg)  负样本索引

        Returns:
            loss: 标量
        """
        # h = W[center], shape: (batch, dim)
        center_emb = self.in_embeddings(center)

        # 正样本得分: dot(W'[pos], h)
        pos_emb = self.out_embeddings(pos_context)  # (batch, dim)
        pos_score = torch.sum(center_emb * pos_emb, dim=1)  # (batch,)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)  # (batch,)

        # 负样本得分: dot(W'[neg], h)
        neg_emb = self.out_embeddings(neg_context)  # (batch, num_neg, dim)
        # (batch, num_neg, dim) × (batch, dim, 1) → (batch, num_neg)
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)  # (batch,)

        return (pos_loss + neg_loss).mean()

    def get_word_vectors(self) -> np.ndarray:
        """返回训练好的词向量 (使用 W 输入嵌入矩阵)"""
        return self.in_embeddings.weight.data.cpu().numpy()


class CBOWNegSampling(nn.Module):
    """
    CBOW with Negative Sampling

    与 Skip-gram 共享相同的两个权重矩阵 W 和 W'。
    区别: 隐藏层 h 是上下文词向量的平均值。

    前向过程:
    1. h = mean(W[context_words])    ← 上下文词向量取平均
    2. 正样本: score = dot(W'[target], h)
    3. 负样本: score = dot(W'[neg], h)
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embed_dim)

        init_range = 0.5 / embed_dim
        self.in_embeddings.weight.data.uniform_(-init_range, init_range)
        self.out_embeddings.weight.data.zero_()

    def forward(self, context: torch.Tensor, target: torch.Tensor,
                neg_samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context:     (batch_size, 2*window)  上下文词索引
            target:      (batch_size,)           目标中心词索引
            neg_samples: (batch_size, num_neg)   负样本索引
        """
        # h = mean(W[context]), shape: (batch, dim)
        context_emb = self.in_embeddings(context)  # (batch, 2*window, dim)
        h = context_emb.mean(dim=1)  # (batch, dim)

        # 正样本
        target_emb = self.out_embeddings(target)  # (batch, dim)
        pos_score = torch.sum(h * target_emb, dim=1)
        pos_loss = -torch.nn.functional.logsigmoid(pos_score)

        # 负样本
        neg_emb = self.out_embeddings(neg_samples)  # (batch, num_neg, dim)
        neg_score = torch.bmm(neg_emb, h.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).sum(dim=1)

        return (pos_loss + neg_loss).mean()

    def get_word_vectors(self) -> np.ndarray:
        return self.in_embeddings.weight.data.cpu().numpy()


# ============================================================
# 4. 训练逻辑
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_skipgram(
    embed_dim: int = 300,
    window_size: int = 5,
    num_neg: int = 5,
    min_count: int = 5,
    max_vocab_size: int = 50000,
    batch_size: int = 512,
    epochs: int = 3,
    initial_lr: float = 0.025,
    save_path: Optional[str] = None,
):
    """
    训练 Skip-gram 模型。

    论文关键超参数:
    - embed_dim: 300 (论文 Table 4/5)
    - window_size: 5~10 (论文 Section 3.2, C=5 或 C=10)
    - initial_lr: 0.025, 线性衰减到 0 (论文 Section 4.2)
    - epochs: 3 (论文 Section 4.2), 大数据集用 1 (Section 4.3)
    - num_neg: 5~20 (论文 [21])
    """
    device = get_device()
    print(f"使用设备: {device}")

    # 加载数据
    words = read_text8()
    vocab = Vocabulary(words, min_count=min_count, max_vocab_size=max_vocab_size)
    word_indices = vocab.encode(words)

    # 高频词下采样
    print("正在对高频词进行下采样...")
    word_indices = vocab.subsample(word_indices)
    print(f"下采样后词数: {len(word_indices):,}")

    # 构建数据集
    dataset = SkipGramDataset(word_indices, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    # 模型
    model = SkipGramNegSampling(vocab.vocab_size, embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,} (2 × {vocab.vocab_size} × {embed_dim})")

    # 优化器: SGD，论文使用 SGD + 线性学习率衰减
    optimizer = optim.SGD(model.parameters(), lr=initial_lr)

    # 学习率线性衰减调度器
    total_steps = len(dataloader) * epochs
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(1e-4, 1.0 - step / total_steps)
    )

    # 训练循环
    print(f"\n开始训练 Skip-gram (epochs={epochs}, lr={initial_lr})...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (centers, contexts) in enumerate(dataloader):
            centers = centers.to(device)
            contexts = contexts.to(device)

            # 负采样
            neg = torch.tensor(
                vocab.negative_sample(len(centers) * num_neg).reshape(-1, num_neg),
                dtype=torch.long, device=device
            )

            loss = model(centers, contexts, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 5000 == 0:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} 完成, 平均 Loss: {avg_loss:.4f}")

    # 保存
    if save_path is None:
        save_path = os.path.join(DATA_DIR, "skipgram_vectors.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "word_vectors": model.get_word_vectors(),
        "word2idx": vocab.word2idx,
        "idx2word": vocab.idx2word,
    }, save_path)
    print(f"\n词向量已保存到 {save_path}")

    return model, vocab


def train_cbow(
    embed_dim: int = 300,
    window_size: int = 4,
    num_neg: int = 5,
    min_count: int = 5,
    max_vocab_size: int = 50000,
    batch_size: int = 512,
    epochs: int = 3,
    initial_lr: float = 0.025,
    save_path: Optional[str] = None,
):
    """
    训练 CBOW 模型。

    论文关键超参数:
    - window_size: 4 (论文 Section 3.1, "four future and four history words")
    - 其他同 Skip-gram
    """
    device = get_device()
    print(f"使用设备: {device}")

    words = read_text8()
    vocab = Vocabulary(words, min_count=min_count, max_vocab_size=max_vocab_size)
    word_indices = vocab.encode(words)

    print("正在对高频词进行下采样...")
    word_indices = vocab.subsample(word_indices)
    print(f"下采样后词数: {len(word_indices):,}")

    dataset = CBOWDataset(word_indices, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    model = CBOWNegSampling(vocab.vocab_size, embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    optimizer = optim.SGD(model.parameters(), lr=initial_lr)
    total_steps = len(dataloader) * epochs
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: max(1e-4, 1.0 - step / total_steps)
    )

    print(f"\n开始训练 CBOW (epochs={epochs}, lr={initial_lr})...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (contexts, targets) in enumerate(dataloader):
            contexts = contexts.to(device)
            targets = targets.to(device)

            neg = torch.tensor(
                vocab.negative_sample(len(targets) * num_neg).reshape(-1, num_neg),
                dtype=torch.long, device=device
            )

            loss = model(contexts, targets, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 5000 == 0:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} 完成, 平均 Loss: {avg_loss:.4f}")

    if save_path is None:
        save_path = os.path.join(DATA_DIR, "cbow_vectors.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "word_vectors": model.get_word_vectors(),
        "word2idx": vocab.word2idx,
        "idx2word": vocab.idx2word,
    }, save_path)
    print(f"\n词向量已保存到 {save_path}")

    return model, vocab


# ============================================================
# 5. 评估: 词向量相似度 + 类比测试
# ============================================================

def most_similar(word: str, word_vectors: np.ndarray, word2idx: dict,
                 idx2word: list, top_k: int = 10) -> List[Tuple[str, float]]:
    """查找最相似的词 (余弦相似度)"""
    if word not in word2idx:
        print(f"'{word}' 不在词表中")
        return []

    idx = word2idx[word]
    vec = word_vectors[idx]
    # 归一化
    norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = word_vectors / norms
    vec_normed = vec / max(np.linalg.norm(vec), 1e-8)

    similarities = normed @ vec_normed
    # 排除自身
    similarities[idx] = -1
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = [(idx2word[i], float(similarities[i])) for i in top_indices]
    return results


def analogy(a: str, b: str, c: str, word_vectors: np.ndarray,
            word2idx: dict, idx2word: list, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    类比测试: a - b + c = ?
    例: king - man + woman = queen
    即 vector("king") - vector("man") + vector("woman") ≈ vector("queen")
    """
    for w in [a, b, c]:
        if w not in word2idx:
            print(f"'{w}' 不在词表中")
            return []

    vec = word_vectors[word2idx[a]] - word_vectors[word2idx[b]] + word_vectors[word2idx[c]]

    norms = np.linalg.norm(word_vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = word_vectors / norms
    vec_normed = vec / max(np.linalg.norm(vec), 1e-8)

    similarities = normed @ vec_normed
    # 排除输入词
    for w in [a, b, c]:
        similarities[word2idx[w]] = -1

    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(idx2word[i], float(similarities[i])) for i in top_indices]


# ============================================================
# 6. 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Word2Vec PyTorch 训练")
    parser.add_argument("--model", type=str, default="skipgram", choices=["skipgram", "cbow"])
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--num_neg", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--max_vocab", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.025)
    args = parser.parse_args()

    if args.model == "skipgram":
        model, vocab = train_skipgram(
            embed_dim=args.embed_dim, window_size=args.window,
            num_neg=args.num_neg, min_count=args.min_count,
            max_vocab_size=args.max_vocab, batch_size=args.batch_size,
            epochs=args.epochs, initial_lr=args.lr,
        )
    else:
        model, vocab = train_cbow(
            embed_dim=args.embed_dim, window_size=args.window,
            num_neg=args.num_neg, min_count=args.min_count,
            max_vocab_size=args.max_vocab, batch_size=args.batch_size,
            epochs=args.epochs, initial_lr=args.lr,
        )

    # 简单测试
    vectors = model.get_word_vectors()
    w2i = vocab.word2idx
    i2w = vocab.idx2word

    print("\n" + "=" * 50)
    print("词向量评估")
    print("=" * 50)

    test_words = ["king", "computer", "university", "city", "good"]
    for word in test_words:
        print(f"\n与 '{word}' 最相似的词:")
        for w, sim in most_similar(word, vectors, w2i, i2w, top_k=8):
            print(f"  {w:20s} {sim:.4f}")

    print("\n类比测试 (a - b + c = ?):")
    analogy_tests = [
        ("king", "man", "woman"),      # → queen
        ("paris", "france", "germany"),  # → berlin
        ("bigger", "big", "small"),      # → smaller
    ]
    for a, b, c in analogy_tests:
        print(f"\n  {a} - {b} + {c} = ?")
        for w, sim in analogy(a, b, c, vectors, w2i, i2w, top_k=5):
            print(f"    {w:20s} {sim:.4f}")
