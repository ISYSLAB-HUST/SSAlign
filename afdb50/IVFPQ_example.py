import faiss
import numpy as np

if __name__=="__main__":

# 设置参数
    dim = 128  # 向量维度
    nlist = 100  # 子簇数量
    m = 16  # 每个子向量的字节数
    nbits = 8  # 每个子量化器的比特数
    n = 10000  # 数据点数量

# 创建随机数据集
    xb = np.random.rand(n, dim).astype('float32')

# 创建量化器
    quantizer = faiss.IndexFlatIP(dim)  # 使用内积

# 创建 IVFPQ 索引
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

# 训练索引
    index.train(xb)  # 训练索引
# 添加数据到索引
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

