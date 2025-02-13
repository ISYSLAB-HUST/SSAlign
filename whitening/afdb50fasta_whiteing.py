import numpy as np
import json

class WhiteningProcessor:
    """
    一个用于增量式计算协方差矩阵，均值向量，和进行白化处理的类
    """
    def __init__(self, mu_filename, W_filename, batch_size=1000):
        self.mu_filename = mu_filename
        self.W_filename = W_filename
        self.batch_size = batch_size
        self.mu = None  # 均值向量
        self.Sigma = None  # 协方差矩阵
        self.total_samples = 0  # 已处理的样本数
        self.d = None  # 向量维度

    def update_mean(self, mu, x_new, n):
        """
        更新均值 mu
        """
        return (n / (n + 1)) * mu + (1 / (n + 1)) * x_new

    def update_covariance(self, Sigma, mu, x_new, mu_new, n):
        """
        更新协方差矩阵 Sigma
        """
        Sigma_n = (n / (n + 1)) * (Sigma + np.outer(mu, mu))
        Sigma_new = Sigma_n + (1 / (n + 1)) * np.outer(x_new, x_new) - np.outer(mu_new, mu_new)
        return Sigma_new

    def compute_kernel_bias_incremental(self, file_paths):
        """
        逐步读取文件，增量式计算均值和协方差矩阵
        """
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as f:
                while True:
                    # 存储当前批次的向量和文件名
                    vectors = []
                    file_names = []

                    # 一次读取 batch_size 个向量（每个向量占两行：文件名 + 向量数据）
                    for _ in range(self.batch_size):
                        file_name_line = f.readline()
                        if not file_name_line:
                            break  # 文件结束

                        if not file_name_line.startswith(">"):
                            print(f"Unexpected format in file: {file_name_line.strip()}")
                            continue  # 跳过格式不正确的行

                        file_name = file_name_line[1:].strip()

                        vector_line = f.readline()
                        if not vector_line:
                            print(f"Missing vector data for file: {file_name}")
                            break  # 文件结束

                        try:
                            vector = np.array(eval(vector_line.strip()))  # 转换为一维数组
                            vectors.append(vector.flatten())  # 确保向量是一维
                            file_names.append(file_name)
                        except (ValueError, SyntaxError) as e:
                            print(f"Failed to parse vector for file: {file_name}\nError: {e}")
                            continue  # 跳过解析失败的向量

                    if len(vectors) == 0:
                        print("No valid vectors in this batch.")
                        break

                    # 转换为 NumPy 数组，便于后续计算
                    vectors = np.array(vectors)

                    # 第一次读取数据时，初始化维度、均值和协方差矩阵
                    if self.d is None:
                        self.d = vectors.shape[1]
                        self.mu = np.zeros(self.d)
                        self.Sigma = np.zeros((self.d, self.d))

                    # 增量更新均值和协方差矩阵
                    for x_new in vectors:
                        self.total_samples += 1
                        mu_new = self.update_mean(self.mu, x_new, self.total_samples - 1)
                        self.Sigma = self.update_covariance(self.Sigma, self.mu, x_new, mu_new, self.total_samples - 1)
                        self.mu = mu_new

        return self.Sigma, self.mu, self.total_samples

    def save_mu_and_W(self, W):
        """
        保存mu和W
        """
        np.save(self.mu_filename, self.mu)
        np.save(self.W_filename, W)
        print(f"mu和W已保存：{self.mu_filename}, {self.W_filename}")

    def load_mu_and_W(self):
        """
        加载均值和白化矩阵
        """
        self.mu = np.load(self.mu_filename)
        W = np.load(self.W_filename)
        return self.mu, W

    def whiten_transform(self, X, W, mu):
        """
        根据事先计算好的均值和白化矩阵，进行whitening处理
        """
        X_centered = X - mu
        X_whitened = np.dot(X_centered, W)
        return X_whitened

    def process_file_incremental(self, input_file):
        """
        增量式计算协方差矩阵和均值，并保存白化矩阵
        """
        # 1. 增量式计算协方差矩阵和均值
        Sigma, mu, total_samples = self.compute_kernel_bias_incremental(input_file)

        print("协方差矩阵和均值构建完毕")

        # 2. 计算白化矩阵
        u, s, vh = np.linalg.svd(Sigma)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))

        # 3. 保存mu和W，以后需要使用
        self.save_mu_and_W(W)

    def process_fasta_file(self, input_file, output_file):
        """
        处理 .fasta 文件，对每个向量进行白化变换，并保存结果
        """
        mu, W = self.load_mu_and_W()

        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            while True:
                file_name_line = f_in.readline()
                if not file_name_line:
                    break  # 文件结束

                if not file_name_line.startswith(">"):
                    continue

                file_name = file_name_line[1:].strip()

                vector_line = f_in.readline()
                if not vector_line:
                    break

                try:
                    vector = np.array(eval(vector_line.strip()))
                    vector = vector.flatten()  # 确保是一维向量

                    # 对向量进行白化变换
                    vector_whitened = self.whiten_transform(vector, W, mu)

                    # 将白化后的向量写入输出文件
                    f_out.write(f">{file_name}\n")
                    f_out.write(f"{vector_whitened.tolist()}\n")
                except Exception as e:
                    continue


if __name__ == "__main__":
    file_paths = [f"/data2/zxc_data/afdb50_combined_fasta/split_fasta_{i}_vector.fasta" for i in range(1, 41)]
    print(file_paths)

    # 设置文件路径
    mu_filename = "data2/zxc_data/afdb50_combined_fasta/whitening/whitening_mu"
    W_filename = "data2/zxc_data/afdb50_combined_fasta/whitening/whitening_W"

    # 初始化 WhiteningProcessor 类
    processor = WhiteningProcessor(mu_filename, W_filename, batch_size=1000)

    # 增量计算协方差矩阵和均值，并保存白化矩阵
    processor.process_file_incremental(file_paths)

    # 处理 .fasta 文件并进行白化变换
    input_file = "/data2/zxc_data/afdb50_combined_fasta/split_fasta_1_vector.fasta"
    output_file = "data2/zxc_data/afdb50_combined_fasta/whitening/output_whitened_vectors.fasta"
    processor.process_fasta_file(input_file, output_file)
