import faiss
import numpy as np

if __name__=="__main__":
    # 设置参数
    dim = 128  # 向量维度
    n = 10000  # 数据点数量

# 创建随机数据集
    xb = np.random.rand(n, dim).astype('float32')


# 创建 IVFPQ 索引
    index = faiss.IndexFlatIP(dim)

# 训练索引
# 添加数据到索引
    faiss.normalize_L2(xb)
    index.add(xb)  # 将数据添加到索引

# 查询向量（随机生成）
    k = 5  # 找到前 K 个邻居
    xq = np.random.rand(5, dim).astype('float32')

# 执行归一化以获得余弦相似度
    faiss.normalize_L2(xq)

# 执行搜索
    distances, indices = index.search(xq, k)

# 输出查询结果
    print("Distances (Inner Product):\n", distances)
    print("Indices:\n", indices)

