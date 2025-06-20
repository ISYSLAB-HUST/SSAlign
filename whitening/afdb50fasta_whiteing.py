import numpy as np
import json

class WhiteningProcessor:
    """
    """
    def __init__(self, mu_filename, W_filename, batch_size=1000):
        self.mu_filename = mu_filename
        self.W_filename = W_filename
        self.batch_size = batch_size
        self.mu = None  # mu
        self.Sigma = None  #  sigma
        self.total_samples = 0  
        self.d = None  # dim

    def update_mean(self, mu, x_new, n):
        """
        update mu
        """
        return (n / (n + 1)) * mu + (1 / (n + 1)) * x_new

    def update_covariance(self, Sigma, mu, x_new, mu_new, n):
        """
        update Sigma
        """
        Sigma_n = (n / (n + 1)) * (Sigma + np.outer(mu, mu))
        Sigma_new = Sigma_n + (1 / (n + 1)) * np.outer(x_new, x_new) - np.outer(mu_new, mu_new)
        return Sigma_new

    def compute_kernel_bias_incremental(self, file_paths):
        """
        """
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            with open(file_path, 'r') as f:
                while True:
                    # 
                    vectors = []
                    file_names = []

                    for _ in range(self.batch_size):
                        file_name_line = f.readline()
                        if not file_name_line:
                            break  # 

                        if not file_name_line.startswith(">"):
                            print(f"Unexpected format in file: {file_name_line.strip()}")
                            continue  # skip

                        file_name = file_name_line[1:].strip()

                        vector_line = f.readline()
                        if not vector_line:
                            print(f"Missing vector data for file: {file_name}")
                            break  # 

                        try:
                            vector = np.array(eval(vector_line.strip()))  # 
                            vectors.append(vector.flatten())  # 
                            file_names.append(file_name)
                        except (ValueError, SyntaxError) as e:
                            print(f"Failed to parse vector for file: {file_name}\nError: {e}")
                            continue  # 

                    if len(vectors) == 0:
                        print("No valid vectors in this batch.")
                        break

                    vectors = np.array(vectors)

                    if self.d is None:
                        self.d = vectors.shape[1]
                        self.mu = np.zeros(self.d)
                        self.Sigma = np.zeros((self.d, self.d))

                    for x_new in vectors:
                        self.total_samples += 1
                        mu_new = self.update_mean(self.mu, x_new, self.total_samples - 1)
                        self.Sigma = self.update_covariance(self.Sigma, self.mu, x_new, mu_new, self.total_samples - 1)
                        self.mu = mu_new

        return self.Sigma, self.mu, self.total_samples

    def save_mu_and_W(self, W):
        """
        
        """
        np.save(self.mu_filename, self.mu)
        np.save(self.W_filename, W)
        print(f"mu和W已保存：{self.mu_filename}, {self.W_filename}")

    def load_mu_and_W(self):
        """
        
        """
        self.mu = np.load(self.mu_filename)
        W = np.load(self.W_filename)
        return self.mu, W

    def whiten_transform(self, X, W, mu):
        """
        """
        X_centered = X - mu
        X_whitened = np.dot(X_centered, W)
        return X_whitened

    def process_file_incremental(self, input_file):
        """
        """
        # 1. 
        Sigma, mu, total_samples = self.compute_kernel_bias_incremental(input_file)


        # 2. 
        u, s, vh = np.linalg.svd(Sigma)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))

        # 3. 
        self.save_mu_and_W(W)

    def process_fasta_file(self, input_file, output_file):
        """
        """
        mu, W = self.load_mu_and_W()

        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            while True:
                file_name_line = f_in.readline()
                if not file_name_line:
                    break  # 

                if not file_name_line.startswith(">"):
                    continue

                file_name = file_name_line[1:].strip()

                vector_line = f_in.readline()
                if not vector_line:
                    break

                try:
                    vector = np.array(eval(vector_line.strip()))
                    vector = vector.flatten()  # 

                    vector_whitened = self.whiten_transform(vector, W, mu)

                    
                    f_out.write(f">{file_name}\n")
                    f_out.write(f"{vector_whitened.tolist()}\n")
                except Exception as e:
                    continue


if __name__ == "__main__":
    file_paths = [f"../afdb50_combined_fasta/split_fasta_{i}_vector.fasta" for i in range(1, 41)]
    print(file_paths)

    mu_filename = "../afdb50_combined_fasta/whitening/whitening_mu"
    W_filename = "../afdb50_combined_fasta/whitening/whitening_W"

    processor = WhiteningProcessor(mu_filename, W_filename, batch_size=1000)

    processor.process_file_incremental(file_paths)

    input_file = "../afdb50_combined_fasta/split_fasta_1_vector.fasta"
    output_file = "../afdb50_combined_fasta/whitening/output_whitened_vectors.fasta"
    processor.process_fasta_file(input_file, output_file)
