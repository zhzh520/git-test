# 导入依赖
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='jieba')
import jieba
import numpy as np
import re
from collections import Counter
import time
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# ===================== JSON序列化工具函数 =====================
class NpEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy类型"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)


# ===================== GPU 加速配置（修复重复打印问题） =====================
# 全局变量控制只打印一次GPU信息
_gpu_info_printed = False


def setup_device():
    """设置设备并只打印一次GPU信息"""
    global _gpu_info_printed
    if not _gpu_info_printed:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name(0)}")
            # 开启cuDNN优化
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # 清空GPU缓存
            torch.cuda.empty_cache()
        else:
            print("警告：未使用GPU，训练速度会较慢！")
        _gpu_info_printed = True
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 初始化设备
device = setup_device()


# ===================== 1. 聊斋志异专用数据处理模块 =====================
def advanced_text_cleaning(text):
    """聊斋文本专用清洗函数 - 保留文言特色词汇"""
    # 1. 移除HTML标签和URL
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 2. 保留中文、文言标点和部分特殊字符（聊斋文本特色）
    text = re.sub(r'[^\u4e00-\u9fa5。，！？；：""''()（）【】《》\s]', '', text)

    # 3. 移除多余空格和换行
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. 聊斋专用停用词过滤
    stop_words = {'的', '了', '在', '是', '我', '你', '他', '她', '它', '们', '就', '都', '而', '及', '与', '等', '和'}
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    return text


def load_and_preprocess_data(file_path):
    """加载和预处理聊斋志异文本数据"""
    sentences = []
    valid_sentences = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # 移除行首数字编号
            content = re.sub(r'^\d+\s*', '', line.strip())

            # 聊斋文本专用清洗
            content = advanced_text_cleaning(content)

            # 优化过滤条件（加快处理速度）
            if 8 < len(content) < 800:
                sentences.append(content)
                unique_chars = len(set(content))
                if unique_chars / len(content) > 0.2:
                    valid_sentences.append(content)

    print(f"原始数据条数: {len(sentences)}")
    print(f"清洗后有效数据条数: {len(valid_sentences)}")
    print(f"数据清洗保留率: {len(valid_sentences) / len(sentences) * 100:.2f}%")

    return valid_sentences


def chinese_tokenize(sentences):
    """聊斋志异专用分词处理（优化速度）"""
    # 添加聊斋专用词典
    liao_zhai_words = [
        '聊斋', '志异', '狐', '妖', '鬼', '神', '仙', '怪', '精', '魂', '魄', '道', '僧', '儒',
        '书生', '秀才', '举人', '进士', '官', '吏', '民', '女', '妇', '翁', '媪', '儿', '童',
        '夜', '梦', '幻', '境', '异', '奇', '怪', '变', '化', '死', '生', '离', '合'
    ]

    # 批量添加词典（优化速度）
    for word in liao_zhai_words:
        jieba.add_word(word)

    tokenized_sentences = []
    stop_words = {'的', '了', '在', '是', '我', '你', '他', '都', '就', '和', '与', '及', '等', '有', '无', '不', '也',
                  '还'}

    # 批量处理（每1000条一批）
    batch_size = 1000
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        for sentence in batch:
            words = jieba.lcut(sentence, cut_all=False)

            # 优化过滤逻辑
            words = [
                word.strip() for word in words
                if len(word.strip()) >= 1
                   and not word.isdigit()
                   and word not in stop_words
                   and not re.match(r'^[a-zA-Z]+$', word)
                   and word not in ['。', '，', '！', '？', '；', '：', '"', "'", '(', ')', '（', '）', '【', '】', '《', '》']
            ]

            if len(words) > 1:
                tokenized_sentences.append(words)

        print(f"已处理 {min(i + batch_size, len(sentences))}/{len(sentences)} 条数据")

    return tokenized_sentences


# ===================== 2. 聊斋词汇分析专用模块 =====================
def analyze_liaozhai_vocabulary(tokenized_corpus):
    """分析聊斋志异词汇统计信息（仅显示纯文字高频词）"""
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_freq = Counter(all_words)

    print("\n===== 聊斋志异词汇统计分析 =====")
    print(f"总词汇量: {len(all_words)}")
    print(f"唯一词汇数: {len(word_freq)}")
    print(f"平均句子长度: {float(np.mean([len(sentence) for sentence in tokenized_corpus])):.2f}")

    # 只显示纯文字高频词
    print("\n===== 聊斋志异核心高频词汇（仅文字） =====")
    filtered_words = [
        (word, freq) for word, freq in word_freq.most_common(50)
        if len(word) >= 1 and re.match(r'^[\u4e00-\u9fa5]+$', word)
           and word not in {'之', '乎', '者', '也', '焉', '哉', '夫', '兮'}
    ]

    # 显示前20个核心高频词
    for i, (word, freq) in enumerate(filtered_words[:20], 1):
        print(f"{i:2d}. {word:<6} 出现频次: {freq:4d}")

    return word_freq


def build_vocab(tokenized_corpus, min_count=8):  # 提高最小词频（提速）
    """构建聊斋志异词汇表（优化速度）"""
    # 批量统计词频
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])

    # 过滤低频词
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}

    # 创建索引映射
    idx_to_word = ['<PAD>', '<UNK>'] + list(vocab.keys())
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

    print(f"\n词汇表构建完成:")
    print(f"词汇表总大小: {len(word_to_idx)}")
    print(f"过滤低频词汇数: {len(word_counts) - len(vocab)}")
    print(f"核心词汇占比: {float(len(vocab) / len(word_counts) * 100):.2f}%")

    return word_to_idx, idx_to_word, vocab


# ===================== 3. 评价指标模块（增强类比推理） =====================
class Word2VecEvaluator:
    """Word2Vec模型评价器（优化计算速度+增强类比推理）"""

    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.word_vectors = word_vectors_dict
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.vocab_size = len(word_to_idx)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    def calculate_intrinsic_metrics(self, sample_size=1000):  # 减小采样量
        """计算内在评价指标（优化速度）"""
        # 1. 词向量质量指标
        high_freq_words = self._get_high_frequency_words(top_n=50)  # 减少参考词数量
        avg_similarity = self._calculate_average_similarity(high_freq_words, sample_size)

        # 2. 聚类评价指标（减少聚类数）
        cluster_metrics = self._calculate_cluster_metrics(n_clusters=10, sample_size=sample_size)

        # 3. 词向量方差
        vector_variance = self._calculate_vector_variance(sample_size)

        # 转换为Python原生类型
        return {
            'average_similarity': float(avg_similarity),
            'silhouette_score': float(cluster_metrics['silhouette']),
            'calinski_harabasz': float(cluster_metrics['calinski_harabasz']),
            'vector_variance': float(vector_variance)
        }

    def _get_high_frequency_words(self, top_n=50):
        """获取高频词"""
        return list(self.word_to_idx.keys())[:top_n]

    def _calculate_average_similarity(self, reference_words, sample_size):
        """优化相似度计算"""
        similarities = []
        sample_words = list(self.word_vectors.keys())[:sample_size]

        # 向量化计算（优化速度）
        ref_vectors = np.array(
            [self.word_vectors[word].numpy() for word in reference_words if word in self.word_vectors])

        for word in sample_words:
            if word in reference_words:
                continue
            if word in self.word_vectors:
                vec = self.word_vectors[word].numpy()
                # 批量计算相似度
                sims = 1 - np.array([cosine(vec, ref_vec) for ref_vec in ref_vectors])
                max_sim = float(sims.max()) if len(sims) > 0 else 0.0
                similarities.append(max_sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def _calculate_cluster_metrics(self, n_clusters=10, sample_size=1000):
        """优化聚类计算"""
        sample_words = list(self.word_vectors.keys())[:sample_size]
        vectors = np.array([self.word_vectors[word].numpy() for word in sample_words])

        # 使用更快的聚类算法
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)  # 减少初始化次数
        labels = kmeans.fit_predict(vectors)

        silhouette = float(silhouette_score(vectors, labels))
        calinski_harabasz = float(calinski_harabasz_score(vectors, labels))

        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski_harabasz
        }

    def _calculate_vector_variance(self, sample_size):
        """计算词向量方差"""
        sample_words = list(self.word_vectors.keys())[:sample_size]
        vectors = np.array([self.word_vectors[word].numpy() for word in sample_words])

        variance_per_dim = np.var(vectors, axis=0)
        avg_variance = float(np.mean(variance_per_dim))

        return avg_variance

    def evaluate_liaozhai_analogy(self):
        """聊斋志异专用类比任务评价（增强版）"""
        # 扩展类比数据集，增加更多聊斋特色类比对
        analogy_pairs = [
            # 人物关系
            ('狐', '女', '鬼', '魂'),
            ('书生', '读书', '道士', '修道'),
            ('官', '府', '民', '家'),
            ('父', '子', '母', '女'),
            ('师', '徒', '君', '臣'),
            # 时空关系
            ('昼', '夜', '生', '死'),
            ('春', '花', '秋', '月'),
            ('山', '石', '水', '波'),
            ('城', '市', '乡', '村'),
            # 属性关系
            ('人', '宅', '狐', '穴'),
            ('刀', '砍', '剑', '刺'),
            ('火', '热', '冰', '冷'),
            ('风', '吹', '雨', '淋'),
            # 状态关系
            ('醒', '梦', '醉', '醒'),
            ('贫', '富', '贱', '贵'),
            ('病', '痛', '伤', '苦')
        ]

        correct = 0
        total = len(analogy_pairs)
        correct_pairs = []
        incorrect_pairs = []

        for a, b, c, d in analogy_pairs:
            if all(word in self.word_vectors for word in [a, b, c, d]):
                vec_a = self.word_vectors[a].numpy()
                vec_b = self.word_vectors[b].numpy()
                vec_c = self.word_vectors[c].numpy()
                vec_d = self.word_vectors[d].numpy()

                # 类比推理核心公式: a - b + c ≈ d
                target_vec = vec_b - vec_a + vec_c  # 修正公式，更符合word2vec类比逻辑

                # 优化相似度计算（使用全部词汇）
                max_sim = -1.0
                best_match = None
                # 采样更多词汇提高准确性
                sample_words = list(self.word_vectors.keys())[:5000]

                for word in sample_words:
                    if word not in [a, b, c]:
                        vec = self.word_vectors[word].numpy()
                        sim = float(1 - cosine(target_vec, vec))
                        if sim > max_sim:
                            max_sim = sim
                            best_match = word

                if best_match == d:
                    correct += 1
                    correct_pairs.append((a, b, c, d, best_match, max_sim))
                else:
                    incorrect_pairs.append((a, b, c, d, best_match, max_sim))
            else:
                incorrect_pairs.append((a, b, c, d, "词汇缺失", 0.0))

        accuracy = float(correct / total) if total > 0 else 0.0
        return {
            'analogy_accuracy': accuracy,
            'correct': int(correct),
            'total': int(total),
            'correct_pairs': correct_pairs,
            'incorrect_pairs': incorrect_pairs,
            'all_pairs': analogy_pairs
        }

    def analogy_inference(self, a, b, c, topn=5):
        """通用类比推理函数，支持自定义输入"""
        """
        类比推理: a : b = c : ?
        返回最可能的topn个结果
        """
        if not all(word in self.word_vectors for word in [a, b, c]):
            missing = [w for w in [a, b, c] if w not in self.word_vectors]
            raise KeyError(f"以下词汇不在词汇表中: {missing}")

        vec_a = self.word_vectors[a].numpy()
        vec_b = self.word_vectors[b].numpy()
        vec_c = self.word_vectors[c].numpy()

        # 类比核心计算: target = b - a + c
        target_vec = vec_b - vec_a + vec_c

        # 计算所有词汇与目标向量的相似度
        similarities = []
        for word in self.word_vectors.keys():
            if word not in [a, b, c]:  # 排除输入词汇
                vec = self.word_vectors[word].numpy()
                sim = float(1 - cosine(target_vec, vec))
                similarities.append((word, sim))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:topn]


# ===================== 4. 数据准备模块（核心提速） =====================
def create_training_data(tokenized_corpus, word_to_idx, vocab, window_size=3, num_negatives=3):
    """创建Word2Vec训练数据（大幅优化速度）"""
    training_data = []
    vocab_size = len(word_to_idx)
    unk_idx = word_to_idx.get('<UNK>', 0)

    # 优化词频计算
    word_counts = np.zeros(vocab_size)
    for word, idx in word_to_idx.items():
        if word in vocab:
            word_counts[idx] = vocab[word]

    # 负采样分布（优化计算）
    word_distribution = np.power(word_counts, 0.75)
    if word_distribution.sum() == 0:
        word_distribution = np.ones_like(word_distribution) / len(word_distribution)
    else:
        word_distribution = word_distribution / word_distribution.sum()

    # 批量处理句子
    for sentence in tokenized_corpus:
        sentence_indices = [word_to_idx.get(word, unk_idx) for word in sentence]

        # 向量化窗口计算
        for i in range(len(sentence_indices)):
            target = sentence_indices[i]
            # 缩小窗口大小
            start = max(0, i - window_size)
            end = min(len(sentence_indices), i + window_size + 1)

            # 批量添加上下文
            context_pairs = [(target, sentence_indices[j]) for j in range(start, end) if j != i]
            training_data.extend(context_pairs)

    print(f"创建训练样本数: {len(training_data)}")
    return training_data, word_distribution


class Word2VecDataset(Dataset):
    """优化后的Dataset（预生成负样本）"""

    def __init__(self, training_data, word_distribution, num_negatives=3):
        self.training_data = training_data
        self.num_negatives = num_negatives
        self.vocab_size = len(word_distribution)

        # 预生成所有负样本（核心提速）
        print("预生成负样本（提速关键）...")
        start_time = time.time()
        total_negatives = len(training_data) * num_negatives

        # 批量生成负样本
        if word_distribution.sum() > 0:
            self.negative_samples = np.random.choice(
                self.vocab_size,
                size=total_negatives,
                p=word_distribution
            ).reshape(len(training_data), num_negatives)
        else:
            self.negative_samples = np.random.randint(
                0, self.vocab_size,
                size=(len(training_data), num_negatives)
            )

        print(f"负样本预生成完成！耗时: {time.time() - start_time:.2f}秒")

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        target, context = self.training_data[idx]
        # 直接使用预生成的负样本（无需实时计算）
        negative_samples = self.negative_samples[idx]

        # 简单过滤（快速）
        negative_samples = [n for n in negative_samples if n != target and n != context]
        # 补充少量样本
        while len(negative_samples) < self.num_negatives:
            negative_samples.append(np.random.randint(0, self.vocab_size))

        return {
            'target': torch.tensor(target, dtype=torch.long),
            'context': torch.tensor(context, dtype=torch.long),
            'negatives': torch.tensor(negative_samples[:self.num_negatives], dtype=torch.long)
        }


# ===================== 5. 模型定义与训练（混合精度） =====================
class Word2VecModel(nn.Module):
    """轻量化Word2Vec模型"""

    def __init__(self, vocab_size, embedding_dim=50):  # 默认50维（提速）
        super(Word2VecModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 快速初始化
        init_range = 0.5 / embedding_dim
        self.target_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, target_word, context_word, negative_words):
        # 向量化计算
        target_embed = self.target_embeddings(target_word)
        context_embed = self.context_embeddings(context_word)
        negative_embed = self.context_embeddings(negative_words)

        # 优化计算逻辑
        positive_score = torch.sum(target_embed * context_embed, dim=1)
        positive_score = torch.clamp(positive_score, max=10, min=-10)

        target_embed_expanded = target_embed.unsqueeze(1)
        negative_score = torch.bmm(negative_embed, target_embed_expanded.transpose(1, 2))
        negative_score = torch.clamp(negative_score.squeeze(2), max=10, min=-10)

        return positive_score, negative_score


def skipgram_loss(positive_score, negative_score):
    """优化损失函数计算"""
    positive_loss = -torch.log(torch.sigmoid(positive_score))
    negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score)), dim=1)
    return (positive_loss + negative_loss).mean()


def train_word2vec(model, dataset, config, run_name="run1"):
    """优化的训练函数（混合精度+大批次）"""
    batch_size = config.get('batch_size', 2048)  # 增大批次
    epochs = config.get('epochs', 5)  # 修改为5轮
    learning_rate = config.get('learning_rate', 0.05)  # 提高学习率
    embedding_dim = config.get('embedding_dim', 50)

    model = model.to(device)

    # 优化DataLoader（核心提速）
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,  # 多进程
        pin_memory=True,  # 固定内存
        prefetch_factor=2,  # 预取数据
        persistent_workers=True if torch.cuda.is_available() else False  # 持久化进程
    )

    # 优化器选择
    optimizer_type = config.get('optimizer', 'Adam')
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 简化学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 混合精度训练（GPU提速关键）
    scaler = GradScaler() if torch.cuda.is_available() else None

    print(f"\n开始训练 {run_name}...")
    print(f"配置: {config}")
    print(f"混合精度训练: {'开启' if scaler else '关闭'}")

    training_log = {
        'losses': [],
        'epoch_times': [],
        'config': config,
        'metrics': {}
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        # 减少打印频率
        print_interval = max(100, len(dataloader) // 10)

        for batch_idx, batch in enumerate(dataloader):
            target_words = batch['target'].to(device, non_blocking=True)  # 非阻塞传输
            context_words = batch['context'].to(device, non_blocking=True)
            negative_words = batch['negatives'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 优化内存

            # 混合精度训练
            if scaler:
                with autocast():
                    positive_score, negative_score = model(target_words, context_words, negative_words)
                    loss = skipgram_loss(positive_score, negative_score)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                positive_score, negative_score = model(target_words, context_words, negative_words)
                loss = skipgram_loss(positive_score, negative_score)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item())

            # 减少打印次数
            if batch_idx % print_interval == 0 and batch_idx > 0:
                avg_batch_loss = float(total_loss / (batch_idx + 1))
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {avg_batch_loss:.4f}")

        scheduler.step()
        avg_loss = float(total_loss / len(dataloader))
        training_log['losses'].append(avg_loss)
        epoch_time = float(time.time() - start_time)
        training_log['epoch_times'].append(epoch_time)

        print(f"Epoch {epoch + 1}/{epochs} 完成 | 平均损失: {avg_loss:.4f} | 时间: {epoch_time:.2f}秒")

    print("\n训练完成！")
    return model, training_log


# ===================== 6. 词向量提取与分析 =====================
def get_word_vectors(model, word_to_idx):
    """快速提取词向量"""
    model.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(word_to_idx)).to(device)
        word_vectors = model.target_embeddings(all_indices).detach().cpu()

    word_vectors_dict = {}
    # 批量构建字典
    for word, idx in word_to_idx.items():
        word_vectors_dict[word] = word_vectors[idx]

    return word_vectors_dict, word_vectors


def save_word_vectors_pt(word_vectors_dict, output_path):
    """保存词向量"""
    torch.save(word_vectors_dict, output_path)
    print(f"词向量已保存到: {output_path}")


def load_word_vectors_pt(input_path):
    """加载词向量"""
    word_vectors_dict = torch.load(input_path)
    print(f"词向量已从 {input_path} 加载")
    return word_vectors_dict


class LiaozhaiWord2VecWrapper:
    """聊斋专用Word2Vec包装器（优化相似度计算）"""

    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.wv = self.WordVectors(word_vectors_dict, word_to_idx, idx_to_word)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    class WordVectors:
        def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
            self.vectors_dict = word_vectors_dict
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
            self.key_to_index = word_to_idx
            self.vectors = np.stack(list(word_vectors_dict.values()))

        def __getitem__(self, word):
            return self.vectors_dict.get(word, None)

        def __contains__(self, word):
            return word in self.vectors_dict

        def similarity(self, word1, word2):
            """优化相似度计算"""
            if word1 not in self.vectors_dict or word2 not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word1} 或 {word2}")

            vec1 = self.vectors_dict[word1].numpy()
            vec2 = self.vectors_dict[word2].numpy()

            norm1 = float(np.linalg.norm(vec1))
            norm2 = float(np.linalg.norm(vec2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))

        def most_similar(self, word, topn=10):
            """优化相似词查找（批量计算）"""
            if word not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word}")

            target_vec = self.vectors_dict[word].numpy()
            # 批量计算相似度（向量化）
            all_vecs = np.array([vec.numpy() for vec in self.vectors_dict.values()])
            all_words = list(self.vectors_dict.keys())

            # 归一化
            target_norm = float(np.linalg.norm(target_vec))
            vecs_norm = np.linalg.norm(all_vecs, axis=1)

            # 避免除零
            mask = (target_norm > 0) & (vecs_norm > 0)
            similarities = np.zeros(len(all_words))

            if mask.any():
                dot_products = np.dot(all_vecs, target_vec)
                similarities[mask] = dot_products[mask] / (target_norm * vecs_norm[mask])

            # 打包并排序
            word_sim_pairs = list(zip(all_words, similarities))
            # 排除自身
            word_sim_pairs = [(w, float(s)) for w, s in word_sim_pairs if w != word]
            # 排序
            word_sim_pairs.sort(key=lambda x: x[1], reverse=True)

            return word_sim_pairs[:topn]

        def domain_similarity_analysis(self, domain_words):
            """聊斋领域相关词相似度分析（优化速度）"""
            print(f"\n===== 聊斋领域词汇相似度分析 =====")
            results = {}

            # 预加载所有向量
            vecs = {}
            for word in domain_words:
                if word in self.vectors_dict:
                    vecs[word] = self.vectors_dict[word].numpy()

            # 批量计算相似度
            for i, word1 in enumerate(domain_words):
                if word1 not in vecs:
                    continue
                results[word1] = {}
                for j, word2 in enumerate(domain_words):
                    if i >= j or word2 not in vecs:
                        continue
                    # 快速计算相似度
                    norm1 = float(np.linalg.norm(vecs[word1]))
                    norm2 = float(np.linalg.norm(vecs[word2]))
                    if norm1 == 0 or norm2 == 0:
                        sim = 0.0
                    else:
                        sim = float(np.dot(vecs[word1], vecs[word2]) / (norm1 * norm2))
                    results[word1][word2] = sim
                    print(f"{word1:<6} - {word2:<6} 相似度: {sim:.4f}")

            return results


# ===================== 7. 主程序（优化配置） =====================
def main():
    # 1. 数据加载和预处理
    file_path = r"D:\Users\huihu\Desktop\2026\damoxing\Word2vec\liaozhai.txt"
    sentences = load_and_preprocess_data(file_path)
    print(f"\n总共加载有效聊斋文本: {len(sentences)} 条")

    # 分词
    tokenized_corpus = chinese_tokenize(sentences)

    # 聊斋专用词汇分析
    word_frequency = analyze_liaozhai_vocabulary(tokenized_corpus)

    # 构建词汇表（提高最小词频）
    word_to_idx, idx_to_word, vocab = build_vocab(tokenized_corpus, min_count=8)

    # 创建训练数据（缩小窗口和负采样）
    training_data, word_distribution = create_training_data(
        tokenized_corpus, word_to_idx, vocab, window_size=3, num_negatives=3
    )

    # 创建数据集
    dataset = Word2VecDataset(training_data, word_distribution, num_negatives=3)

    # 优化的训练配置（5组不同实验配置）
    configs = [
        {
            "name": "实验1-基础配置",
            "embedding_dim": 50,
            "batch_size": 2048,
            "epochs": 5,  # 5轮训练
            "learning_rate": 0.05,
            "optimizer": "Adam"
        },
        {
            "name": "实验2-更大维度",
            "embedding_dim": 100,
            "batch_size": 2048,
            "epochs": 5,
            "learning_rate": 0.05,
            "optimizer": "Adam"
        },
        {
            "name": "实验3-SGD优化器",
            "embedding_dim": 50,
            "batch_size": 2048,
            "epochs": 5,
            "learning_rate": 0.1,
            "optimizer": "SGD"
        },
        {
            "name": "实验4-AdamW优化器",
            "embedding_dim": 50,
            "batch_size": 2048,
            "epochs": 5,
            "learning_rate": 0.001,
            "optimizer": "AdamW"
        },
        {
            "name": "实验5-更大批次",
            "embedding_dim": 50,
            "batch_size": 4096,
            "epochs": 5,
            "learning_rate": 0.05,
            "optimizer": "Adam"
        }
    ]

    # 存储实验结果
    all_results = {}

    # 多轮训练和评估（5次实验）
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"开始训练: {config['name']}")
        print(f"{'=' * 60}")

        # 创建模型
        model = Word2VecModel(vocab_size=len(word_to_idx), embedding_dim=config['embedding_dim'])

        # 训练模型
        trained_model, training_log = train_word2vec(
            model, dataset, config, run_name=config['name']
        )

        # 提取词向量
        word_vectors_dict, all_vectors = get_word_vectors(trained_model, word_to_idx)

        # 保存词向量
        save_path = f"liaozhai_word_vectors_{config['name']}.pt"
        save_word_vectors_pt(word_vectors_dict, save_path)

        # 创建评价器
        evaluator = Word2VecEvaluator(word_vectors_dict, word_to_idx, idx_to_word)

        # 计算内在评价指标（减小采样量）
        intrinsic_metrics = evaluator.calculate_intrinsic_metrics(sample_size=1000)

        # 聊斋专用类比任务评价
        analogy_metrics = evaluator.evaluate_liaozhai_analogy()

        # 保存结果（确保所有值都是Python原生类型）
        all_results[config['name']] = {
            'config': config,
            'training_log': {
                'losses': [float(l) for l in training_log['losses']],
                'epoch_times': [float(t) for t in training_log['epoch_times']],
                'config': training_log['config'],
                'metrics': training_log['metrics']
            },
            'intrinsic_metrics': intrinsic_metrics,
            'analogy_metrics': analogy_metrics,
            'save_path': save_path
        }

        # 打印评价结果
        print(f"\n{config['name']} 评价结果:")
        print("1. 内在评价指标:")
        for metric, value in intrinsic_metrics.items():
            print(f"   {metric}: {value:.4f}")

        print("2. 聊斋类比任务评价:")
        print(f"   准确率: {analogy_metrics['analogy_accuracy']:.4f}")
        print(f"   正确数/总数: {analogy_metrics['correct']}/{analogy_metrics['total']}")

        # 打印部分类比推理结果
        print("3. 类比推理示例:")
        if analogy_metrics['correct_pairs']:
            print("   正确示例:")
            for pair in analogy_metrics['correct_pairs'][:3]:
                a, b, c, d, match, sim = pair
                print(f"     {a}:{b} = {c}:{d} (预测: {match}, 相似度: {sim:.4f})")
        if analogy_metrics['incorrect_pairs']:
            print("   错误示例:")
            for pair in analogy_metrics['incorrect_pairs'][:2]:
                a, b, c, d, match, sim = pair
                print(f"     {a}:{b} = {c}:{d} (预测: {match}, 相似度: {sim:.4f})")

    # 结果汇总
    print(f"\n{'=' * 60}")
    print("聊斋志异Word2Vec训练结果汇总（5次实验）")
    print(f"{'=' * 60}")

    # 打印汇总表
    print(f"{'配置名称':<20} {'轮廓系数':<10} {'CH指数':<10} {'类比准确率':<10}")
    print("-" * 50)
    for config_name, results in all_results.items():
        silhouette = results['intrinsic_metrics']['silhouette_score']
        ch_index = results['intrinsic_metrics']['calinski_harabasz']
        analogy_acc = results['analogy_metrics']['analogy_accuracy']
        print(f"{config_name:<20} {silhouette:<10.4f} {ch_index:<10.2f} {analogy_acc:<10.4f}")

    # 快速绘制损失曲线
    plt.figure(figsize=(10, 6))
    for config_name, results in all_results.items():
        losses = results['training_log']['losses']
        plt.plot(range(1, len(losses) + 1), losses, label=config_name, marker='o', markersize=4)

    plt.xlabel('训练轮数（5轮）')
    plt.ylabel('平均损失')
    plt.title('聊斋志异Word2Vec训练损失曲线（5次实验对比）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('liaozhai_training_loss_5experiments.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 保存结果到JSON（使用自定义编码器）
    serializable_results = {}
    for config_name, results in all_results.items():
        serializable_results[config_name] = {
            'config': results['config'],
            'intrinsic_metrics': results['intrinsic_metrics'],
            'analogy_metrics': {
                'accuracy': results['analogy_metrics']['analogy_accuracy'],
                'correct': results['analogy_metrics']['correct'],
                'total': results['analogy_metrics']['total']
            },
            'final_loss': float(results['training_log']['losses'][-1]) if results['training_log']['losses'] else None,
            'total_training_time': float(sum(results['training_log']['epoch_times']))
        }

    # 使用自定义编码器保存JSON
    with open('liaozhai_word2vec_5experiments_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4, cls=NpEncoder)

    print(f"\n实验结果已保存到: liaozhai_word2vec_5experiments_results.json")
    print(f"损失曲线图已保存到: liaozhai_training_loss_5experiments.png")

    # 使用最佳模型进行聊斋词汇分析
    best_config = max(all_results.items(), key=lambda x: x[1]['analogy_metrics']['analogy_accuracy'])
    print(f"\n最佳模型: {best_config[0]} (类比准确率: {best_config[1]['analogy_metrics']['analogy_accuracy']:.4f})")

    # 加载最佳模型
    best_word_vectors = load_word_vectors_pt(best_config[1]['save_path'])
    w2v_model = LiaozhaiWord2VecWrapper(best_word_vectors, word_to_idx, idx_to_word)

    # 创建最佳模型的评价器，用于类比推理
    best_evaluator = Word2VecEvaluator(best_word_vectors, word_to_idx, idx_to_word)

    # ===================== 聊斋志异专用词向量分析 =====================
    print("\n" + "=" * 60)
    print("聊斋志异 - 词向量分析与类比推理")
    print("=" * 60)

    # 1. 核心词汇相似词分析
    print("\n【核心词汇相似词分析】")
    core_words = ['狐', '鬼', '仙', '书生', '道', '梦']
    for word in core_words:
        if word in w2v_model.wv:
            similar_words = w2v_model.wv.most_similar(word, topn=10)
            print(f"\n与'{word}'最相似的10个词：")
            for i, (similar, score) in enumerate(similar_words, 1):
                print(f"  {i:2d}. {similar:<6} 相似度: {score:.4f}")
        else:
            print(f"\n'{word}'不在词汇表中")

    # 2. 聊斋领域相关词相似度分析
    print("\n【聊斋领域词汇相似度分析】")
    domain_words = ['狐', '鬼', '仙', '妖', '精']
    w2v_model.wv.domain_similarity_analysis(domain_words)

    # 3. 自定义类比推理示例
    print("\n【自定义类比推理示例】")
    analogy_tasks = [
        ('书生', '读书', '道士'),
        ('狐', '女', '鬼'),
        ('昼', '夜', '生'),
        ('官', '府', '民')
    ]

    for a, b, c in analogy_tasks:
        try:
            results = best_evaluator.analogy_inference(a, b, c, topn=3)
            print(f"\n{a}:{b} = {c}:?")
            for i, (word, sim) in enumerate(results, 1):
                print(f"  {i}. {word:<6} 相似度: {sim:.4f}")
        except KeyError as e:
            print(f"\n{a}:{b} = {c}:? - 错误: {e}")


if __name__ == "__main__":
    # 记录总运行时间
    start_total = time.time()
    main()
    print(f"\n总运行时间: {time.time() - start_total:.2f}秒")