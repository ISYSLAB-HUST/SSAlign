import numpy as np
import json
import logging


class WhiteningModel:
    def __init__(self, vector_index: int, batch_size: int, mu_filename: str, W_filename: str):
        """
        初始化 WhiteningProcessor 类
        """
        self.vector_index = vector_index
        self.batch_size = batch_size
        self.mu_filename = mu_filename
        self.W_filename = W_filename

        self.mu = None  # 均值向量
        self.Sigma = None  # 协方差矩阵
        self.total_samples = 0  # 已处理的样本数

    def update_mean(self, mu: np.ndarray, x_new: np.ndarray, n: int) -> np.ndarray:
        """更新均值向量"""
        return (n / (n + 1)) * mu + (1 / (n + 1)) * x_new

    def update_covariance(self, Sigma: np.ndarray, mu: np.ndarray, x_new: np.ndarray, mu_new: np.ndarray,
                          n: int) -> np.ndarray:
        """更新协方差矩阵"""
        Sigma_n = (n / (n + 1)) * (Sigma + np.outer(mu, mu))
        Sigma_new = Sigma_n + (1 / (n + 1)) * np.outer(x_new, x_new) - np.outer(mu_new, mu_new)
        return Sigma_new

    def compute_kernel_bias_incremental(self, file_path: str):
        """增量式计算均值和协方差矩阵"""
        d = None  # 向量维度

        with open(file_path, 'r') as f:
            while True:
                lines = [f.readline().strip() for _ in range(self.batch_size)]
                lines = [line for line in lines if line]  # 移除空行

                if not lines:
                    break

                vectors = []
                for line in lines:
                    try:
                        parts = line.strip().split(",")
                        vector_str = ",".join(parts[self.vector_index:])
                        vector = np.array(json.loads(f'[{vector_str}]'))  # 使用 json.loads() 替代 eval()
                        flat_vector = vector.flatten()
                        vectors.append(flat_vector)
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Failed to parse line: {line}, Error: {e}")
                        continue  # 忽略无法解析的行

                if vectors:
                    vectors = np.array(vectors)

                    # 初始化均值向量和协方差矩阵
                    if d is None:
                        d = vectors.shape[1]
                        self.mu = np.zeros(d)
                        self.Sigma = np.zeros((d, d))

                    # 更新均值和协方差矩阵
                    for x_new in vectors:
                        self.total_samples += 1
                        mu_new = self.update_mean(self.mu, x_new, self.total_samples - 1)
                        self.Sigma = self.update_covariance(self.Sigma, self.mu, x_new, mu_new, self.total_samples - 1)
                        self.mu = mu_new

        return self.Sigma, self.mu, self.total_samples

    def whiten_transform(self, X: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """执行白化变换"""
        X_centered = X - mu
        return np.dot(X_centered, W)

    def save_mu_and_W(self, W: np.ndarray):
        """保存mu和W"""
        np.save(self.mu_filename, self.mu)
        np.save(self.W_filename, W)
        logging.info(f"mu 和 W 已保存：{self.mu_filename}, {self.W_filename}")

    def process_file_incremental(self, input_file: str, output_file: str):
        """处理文件并进行白化变换"""
        # 1. 增量式计算协方差矩阵和均值
        Sigma, mu, total_samples = self.compute_kernel_bias_incremental(input_file)
        logging.info("协方差矩阵和均值构建完毕")

        # 2. 计算白化矩阵
        u, s, vh = np.linalg.svd(Sigma)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))

        # 3. 保存mu和W
        self.save_mu_and_W(W)

        # 4. 逐批处理文件进行白化并保存结果
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            while True:
                lines = [f_in.readline().strip() for _ in range(self.batch_size)]
                lines = [line for line in lines if line]

                if not lines:
                    break

                vectors = []
                file_names = []
                for line in lines:
                    try:
                        parts = line.strip().split(",")
                        file_name = parts[0]
                        vector_str = ",".join(parts[self.vector_index:])
                        vector = np.array(json.loads(f'[{vector_str}]'))  # 安全地解析向量
                        flat_vector = vector.flatten()
                        file_names.append(file_name)
                        vectors.append(flat_vector)
                    except (json.JSONDecodeError, ValueError):
                        continue  # 忽略无法解析的行

                if vectors:
                    vectors = np.array(vectors)

                    # 白化处理
                    whitened_vectors = self.whiten_transform(vectors, W, self.mu)

                    # 将结果写入文件
                    for file_name, vec in zip(file_names, whitened_vectors):
                        vec_str = "[[" + ",".join(map(str, vec)) + "]]"
                        f_out.write(f"{file_name},{vec_str}\n")




# 主函数
if __name__ == "__main__":
    # 初始化 WhiteningProcessor
    vector_index = 2  # 向量在数据中的索引（例如第三列）
    batch_size = 1000  # 每次处理的批次大小
    mu_filename = "/data/foldseek_database/sp_cif_file/sp_whitening_mu"
    W_filename = "/data/foldseek_database/sp_cif_file/sp_whitening_W"

    processor = WhiteningModel(vector_index, batch_size, mu_filename, W_filename)

    # 文件路径
    input_file = '/data/foldseek_database/sp_cif_file/swissprot_cif_v4_files_results'
    output_file = '/data/foldseek_database/sp_cif_file/swissprot_cif_v4_files_results_whitening'

    # 处理文件
    processor.process_file_incremental(input_file, output_file)
