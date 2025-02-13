import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import random


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 一个输入，一个输出
        self.linear = self.linear.double()

    def forward(self, x):
        return self.linear(x)


def process_all_files_torch(basenames,dim,topk, device,cos_threshold):
    # 用于存储合并后的数据
    combined_data = []

    # 逐个读取文件并合并数据
    for basename in basenames:
        csv_file_path = f"../data/result/Swissport/SSAlign/SVD{dim}/{basename}.result"
        df = pd.read_csv(csv_file_path)

        df_sorted = df.sort_values(by='Cosine_Similarity', ascending=False).head(topk)

        # 只对 大于 0.2的拟合
        df_filtered = df_sorted[df_sorted['Cosine_Similarity'] >= cos_threshold]

        # 提取需要的列
        cosine_similarity = df_filtered['Cosine_Similarity'].values
        tm_score1 = df_filtered['TM-Score1'].values
        tm_score2 = df_filtered['TM-Score2'].values
        rmsd_score = df_filtered['RMSD'].values


        avg_tmscore = (tm_score1 + tm_score2) / 2

        combined_score = avg_tmscore

        print(len(df_filtered))

        # 合并数据
        combined_data.append({
            'Cosine_Similarity': cosine_similarity,
            'Avg_TM_Score': avg_tmscore,
            'combined_score': combined_score
        })



    # 将所有文件的数据合并成一个 DataFrame
    combined_df = pd.concat([pd.DataFrame(data) for data in combined_data], ignore_index=True)
    combined_df_avg = combined_df.groupby('Cosine_Similarity').mean().reset_index()

    print(len(combined_df_avg))

    # 将数据转换为 PyTorch 张量
    X = torch.tensor(combined_df_avg['Cosine_Similarity'].values.reshape(-1, 1), dtype=torch.float64).to(device)
    y = torch.tensor(combined_df_avg['combined_score'].values, dtype=torch.float64).to(device)

    return X, y


def train_linear_model(X, y, device,batch_size=32):
    # 创建并训练模型
    model = LinearModel().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 训练模型
    epochs = 1000
    num_batches = len(X) // batch_size
    losses = []  # 用于记录每个 epoch 的平均损失


    for epoch in range(epochs):
        model.train()

        epoch_loss = 0
        for i in range(num_batches):
            # 获取当前小批量数据
            start = i * batch_size
            end = start + batch_size
            X_batch = X[start:end]
            y_batch = y[start:end]

            optimizer.zero_grad()  # 清除梯度
            y_pred = model(X_batch).squeeze()  #  前向传播并去掉多余维度
            loss = criterion(y_pred, y_batch)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)  # 保存每个 epoch 的平均损失
        # 打印每个epoch的平均损失
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / num_batches}')

    # 返回模型参数（斜率和截距）
    slope = model.linear.weight.item()
    intercept = model.linear.bias.item()

    print(f"Trained Model: Slope: {slope}, Intercept: {intercept}")

    return model, slope, intercept,losses


def plot_loss_curve(losses,dim,topk,cos_threshold):
    # 绘制损失曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(losses) + 1), losses, color='blue', label='Training Loss')

    # 添加标题和标签
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.grid(True)
    plt.legend()

    # 保存图片
    plt.savefig(f"{dim}_{topk}_{cos_threshold}_losses.png")
    plt.close()


def plot_model_results(X, y, model, slope, intercept,dim,topk,cos_threshold):
    # 绘制结果
    plt.figure(figsize=(8, 6))

    # 绘制数据点
    # plt.scatter(X.cpu().numpy(), y.cpu().numpy(), color='blue', alpha=0.6, edgecolors='w', s=100)

    # 绘制拟合的回归直线
    y_fit = model(X).cpu().detach().numpy()
    plt.plot(X.cpu().numpy(), y_fit, color='red', label=f'Fitted Line: y = {slope}x + {intercept}')

    # 添加标题和标签
    plt.title('Linear Regression Fit: Cosine Similarity vs Combined Score', fontsize=14, fontweight='bold')
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Combined Score', fontsize=12)

    plt.grid(True)
    plt.legend()

    # 保存图片
    plt.savefig(
        f"{dim}_{topk}_{cos_threshold}_linear.png")
    plt.close()


"""
利用这 100 个文件来拟合的直线
y=1.6368176267837866x+ 0.2547128724857352
"""
def main(dim,topk,cos_threshold):

    with open("100filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 去掉空行和多余空格

    # 检查是否有 GPU 可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    device = "cpu"

    # 获取数据
    X, y = process_all_files_torch(basenames, dim,topk,device,cos_threshold)

    # 训练线性模型
    model, slope, intercept, losses = train_linear_model(X, y, device)

    # 绘制结果
    plot_model_results(X, y, model, slope, intercept,dim,topk,cos_threshold)

    # 绘制损失曲线
    plot_loss_curve(losses,dim,topk,cos_threshold)






if __name__ == "__main__":

    #main(SVD1280,2000,0.2)
    main(512,2000,0.3)
    main(256,2000,0.45)
    main(128,2000,0.6)
    main(64,2000,0.7)

