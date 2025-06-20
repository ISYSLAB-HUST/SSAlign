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

        self.mu = None  # 
        self.Sigma = None  # 
        self.total_samples = 0  # 

    def update_mean(self, mu: np.ndarray, x_new: np.ndarray, n: int) -> np.ndarray:
        
        return (n / (n + 1)) * mu + (1 / (n + 1)) * x_new

    def update_covariance(self, Sigma: np.ndarray, mu: np.ndarray, x_new: np.ndarray, mu_new: np.ndarray,
                          n: int) -> np.ndarray:
        Sigma_n = (n / (n + 1)) * (Sigma + np.outer(mu, mu))
        Sigma_new = Sigma_n + (1 / (n + 1)) * np.outer(x_new, x_new) - np.outer(mu_new, mu_new)
        return Sigma_new

    def compute_kernel_bias_incremental(self, file_path: str):
        d = None  # 

        with open(file_path, 'r') as f:
            while True:
                lines = [f.readline().strip() for _ in range(self.batch_size)]
                lines = [line for line in lines if line]  # 

                if not lines:
                    break

                vectors = []
                for line in lines:
                    try:
                        parts = line.strip().split(",")
                        vector_str = ",".join(parts[self.vector_index:])
                        vector = np.array(json.loads(f'[{vector_str}]'))  #
                        flat_vector = vector.flatten()
                        vectors.append(flat_vector)
                    except (json.JSONDecodeError, ValueError) as e:
                        logging.error(f"Failed to parse line: {line}, Error: {e}")
                        continue  # 

                if vectors:
                    vectors = np.array(vectors)

                    if d is None:
                        d = vectors.shape[1]
                        self.mu = np.zeros(d)
                        self.Sigma = np.zeros((d, d))

                    for x_new in vectors:
                        self.total_samples += 1
                        mu_new = self.update_mean(self.mu, x_new, self.total_samples - 1)
                        self.Sigma = self.update_covariance(self.Sigma, self.mu, x_new, mu_new, self.total_samples - 1)
                        self.mu = mu_new

        return self.Sigma, self.mu, self.total_samples

    def whiten_transform(self, X: np.ndarray, W: np.ndarray, mu: np.ndarray) -> np.ndarray:
        X_centered = X - mu
        return np.dot(X_centered, W)

    def save_mu_and_W(self, W: np.ndarray):
        np.save(self.mu_filename, self.mu)
        np.save(self.W_filename, W)
        logging.info(f"mu 和 W 已保存：{self.mu_filename}, {self.W_filename}")

    def process_file_incremental(self, input_file: str, output_file: str):
        # 1. 
        Sigma, mu, total_samples = self.compute_kernel_bias_incremental(input_file)
        logging.info("协方差矩阵和均值构建完毕")

        # 2. 
        u, s, vh = np.linalg.svd(Sigma)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))

        # 3. 
        self.save_mu_and_W(W)

        # 4. 
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
                        vector = np.array(json.loads(f'[{vector_str}]'))  # 
                        flat_vector = vector.flatten()
                        file_names.append(file_name)
                        vectors.append(flat_vector)
                    except (json.JSONDecodeError, ValueError):
                        continue  # 

                if vectors:
                    vectors = np.array(vectors)

                    # 
                    whitened_vectors = self.whiten_transform(vectors, W, self.mu)

                    # 
                    for file_name, vec in zip(file_names, whitened_vectors):
                        vec_str = "[[" + ",".join(map(str, vec)) + "]]"
                        f_out.write(f"{file_name},{vec_str}\n")




# 
if __name__ == "__main__":
    vector_index = 2  # 
    batch_size = 1000  # 
    mu_filename = "/data/foldseek_database/sp_cif_file/sp_whitening_mu"
    W_filename = "/data/foldseek_database/sp_cif_file/sp_whitening_W"

    processor = WhiteningModel(vector_index, batch_size, mu_filename, W_filename)

    # 
    input_file = '/data/foldseek_database/sp_cif_file/swissprot_cif_v4_files_results'
    output_file = '/data/foldseek_database/sp_cif_file/swissprot_cif_v4_files_results_whitening'

    # 
    processor.process_file_incremental(input_file, output_file)
