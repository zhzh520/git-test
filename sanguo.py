import jieba
import re
import gensim
from gensim.models import Word2Vec


# 定义函数：读取中文文本文件并转换为句子列表（兼容jieba所有版本）
def read_chinese_file_to_sentences(file_path):
    # 初始化句子列表
    sentences = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 按行读取文本，逐行处理（避免一次性读取大文件内存溢出）
            for line in file:
                # 去除每行首尾空白字符（换行、空格等）
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # 正则清洗：保留中文、数字、常用标点，去除特殊符号
                line = re.sub(r'[^\u4e00-\u9fff\d，。！？；：""''()（）《》]', '', line)

                # 按中文句号/感叹号/问号分句（符合中文阅读习惯）
                for sent in re.split(r'[。！？]', line):
                    sent = sent.strip()
                    if len(sent) < 2:  # 跳过过短的无效句子
                        continue
                    # 兼容所有jieba版本：用list(jieba.cut())替代jieba.lcut()
                    words = list(jieba.cut(sent))
                    sentences.append(words)
        return sentences
    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}，请检查路径是否正确！")
        return []
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return []


# 修复路径转义问题（原始字符串）
file_path = r"D:\Users\huihu\Desktop\2026\damoxing\W2V\sanguoyanyi.txt"
# 调用函数获取分词后的句子列表
sentences = read_chinese_file_to_sentences(file_path)

# 检查是否成功获取句子
if not sentences:
    print("未获取到有效文本数据，无法训练模型！")
else:
    print(f"成功加载 {len(sentences)} 个句子，开始训练Word2Vec模型...")

    # 训练Word2Vec模型（优化参数）
    model = Word2Vec(
        sentences,
        vector_size=100,  # 词向量维度
        window=8,  # 上下文窗口大小
        min_count=5,  # 忽略出现次数少于5的词
        workers=4,  # 并行训练线程数
        epochs=10,  # 训练轮数
        sg=0  # CBOW算法（适合小文本）
    )

    # 保存模型
    model.save("sanguo_w2v.model")
    print("模型已保存为 sanguo_w2v.model")

    # 加载训练好的模型
    loaded_model = Word2Vec.load("sanguo_w2v.model")

    # 1. 找出与“刘备”最相似的词（异常处理）
    try:
        similar_words = loaded_model.wv.most_similar('刘备', topn=10)
        print("\n=== 与“刘备”语义最相似的词 ===")
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
    except KeyError:
        print("\n错误：“刘备”未出现在训练词典中，请检查文本是否包含该词！")

    # 2. 类比推理：修正逻辑（刘备 - 张飞 + 关羽）
    try:
        analogy_words = loaded_model.wv.most_similar(
            positive=['刘备', '关羽'],  # 加的词
            negative=['张飞'],  # 减的词
            topn=10
        )
        print("\n=== 类比推理：刘备 - 张飞 + 关羽 ===")
        for word, analogy in analogy_words:
            print(f"{word}: {analogy:.4f}")
    except KeyError as e:
        print(f"\n错误：类比推理时缺少词 {e}，请检查文本是否包含该词！")