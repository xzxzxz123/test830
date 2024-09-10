import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. 读取txt文件中的内容
with open("D:/lab/语料.txt", 'r', encoding='utf-8') as file:
    texts = file.readlines()

# 2. 自定义中文停用词表（可以根据实际情况扩展）
stopwords = set(['的', '了', '和', '是', '在', '我', '有', '也', '不', '就', '人', '都'])

# 3. 中文分词函数，去除停用词
def chinese_tokenizer(text):
    return [word for word in jieba.cut(text) if word not in stopwords and word.strip()]

# 将中文文本分词处理
tokenized_texts = [' '.join(chinese_tokenizer(text)) for text in texts]

# 4. 使用TF-IDF进行文本向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_texts)

# 5. K-means 聚类
num_clusters = 5  # 设置聚类的数量，可根据实际数据调整
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X)

# 获取每个文本的聚类标签
labels = kmeans.labels_

# 6. 可视化聚类结果 (PCA)
pca = PCA(n_components=2)
scatter_plot_points = pca.fit_transform(X.toarray())

colors = ['r', 'b', 'c', 'y', 'm']
cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

# 绘制带有颜色的散点图
for i in range(len(colors)):
    # 获取属于第 i 个聚类的点
    cluster_points = scatter_plot_points[kmeans.labels_ == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=cluster_labels[i])

plt.title("K-means Clustering Visualization")
plt.legend()
plt.savefig('kmeans_clustering.png')
plt.show()

# 7. 提取每个聚类的关键词
terms = vectorizer.get_feature_names_out()
def get_top_keywords(cluster_centers, terms, n=5):
    keywords = []
    for cluster in cluster_centers:
        # 获取每个聚类中心点的前 n 个关键词
        top_indices = cluster.argsort()[-n:]
        keywords.append([terms[i] for i in top_indices])
    return keywords

# 关键词提取
keywords_per_cluster = get_top_keywords(kmeans.cluster_centers_, terms)

# 输出每个聚类的关键词
for i, keywords in enumerate(keywords_per_cluster):
    print(f"Cluster {i}: {', '.join(keywords)}")
