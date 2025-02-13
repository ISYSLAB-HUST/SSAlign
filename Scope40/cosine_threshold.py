import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse



"""
cos_threshold 阈值
"""


def cos_test(basenames, dim,topk,cos_threshold):
    # 用于存储合并后的数据
    combined_data = []

    # 逐个读取文件并合并数据
    for basename in basenames:
        # File1,Aligned Length,RMSD,Seq_ID,length_squeue,Avg_TM_Score,Cosine_Similarity,Length,Identity,Similarity,Gaps,Score
        csv_file_path = f"../data/result/Scope40/SSAlign/SVD{dim}/bio_{basename}_lower_global.csv"

        df = pd.read_csv(csv_file_path)

        df_sorted = df.sort_values(by='Cosine_Similarity', ascending=False).head(topk)

        # 提取需要的列并计算 combined_score
        cosine_similarity = df_sorted['Cosine_Similarity']
        avg_tmscore = df_sorted['Avg_TM_Score']
        combined_score = avg_tmscore


        combined_data.append({
            'Cosine_Similarity': cosine_similarity,
            'Avg_TM_Score': avg_tmscore,
            'combined_score': combined_score
        })

    combined_df = pd.concat([pd.DataFrame(data) for data in combined_data], ignore_index=True)

    # 筛选出 Cosine_Similarity < 0.2 的数据
    filtered_df = combined_df[combined_df['Cosine_Similarity'] < cos_threshold]

    count = filtered_df.shape[0]
    #print(f"Cosine_Similarity < {cos_threshold} 的数据数量: {count}")

    #print(f"avg_tmscore < 0.5 的数据数量: {combined_df[combined_df['Avg_TM_Score'] < 0.5].shape[0]}")

    # 小于阈值，分数高，漏选
    filtered_1 = combined_df[(combined_df['Cosine_Similarity'] < cos_threshold) & (combined_df['Avg_TM_Score'] >= 0.5)]
    count_1 = len(filtered_1)
    #print(f"Cosine_Similarity < {cos_threshold} and Avg_TM_Score > 0.5 的数据量: {count_1}")

    # 小于阈值，分数低，不该选
    filtered_2 = combined_df[(combined_df['Cosine_Similarity'] < cos_threshold) & (combined_df['Avg_TM_Score'] < 0.5)]
    count_2 = len(filtered_2)
    #print(f"Cosine_Similarity < {cos_threshold} and Avg_TM_Score < 0.5 的数据量: {count_2}")

    # 大于阈值，分数高，选对了
    filtered_3 = combined_df[(combined_df['Cosine_Similarity'] >= cos_threshold) & (combined_df['Avg_TM_Score'] >= 0.5)]
    count_3 = len(filtered_3)
    #print(f"Cosine_Similarity >  {cos_threshold} and Avg_TM_Score > 0.5 的数据量: {count_3}")

    # 大于阈值，分数低，选错了
    filtered_4 = combined_df[(combined_df['Cosine_Similarity'] >=  cos_threshold) & (combined_df['Avg_TM_Score'] < 0.5)]
    count_4 = len(filtered_4)
    #print(f"Cosine_Similarity >  {cos_threshold} and Avg_TM_Score < 0.5 的数据量:  {count_4}")

    # 准确率
    precision = count_3 / (count_3 + count_4) if count_3 + count_4 > 0 else 0

    # 召回率
    recall = count_3 / (count_3 + count_1) if count_3 + count_1 > 0 else 0

    # f1分数
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def main(dim,topk):

    with open("random_filenames.txt", 'r') as file:
        basenames = [line.strip() for line in file if line.strip()]

    # cos_test(basenames, SVD1280,1000,0.2)

    thresholds = np.arange(0.1, 1, 0.001)  # 步长更小
    precisions = []
    recalls = []
    f1_scores = []

    # 计算每个阈值下的准确率和召回率
    for threshold in thresholds:
        precision, recall, f1_score = cos_test(basenames, dim,topk,threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        # 绘制准确率和召回率的变化曲线
        # 绘制曲线
    plt.plot(thresholds, precisions, label="Precision", color="blue", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", color="green", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1-Score", color="red", linewidth=2)

    # 添加端点标注
    plt.text(thresholds[0], precisions[0], f"{precisions[0]}", color="blue", fontsize=10, ha="right")
    plt.text(thresholds[-1], precisions[-1], f"{precisions[-1]}", color="blue", fontsize=10, ha="left")
    plt.text(thresholds[0], recalls[0], f"{recalls[0]}", color="green", fontsize=10, ha="right")
    plt.text(thresholds[-1], recalls[-1], f"{recalls[-1]}", color="green", fontsize=10, ha="left")
    plt.text(thresholds[0], f1_scores[0], f"{f1_scores[0]}", color="red", fontsize=10, ha="right")
    plt.text(thresholds[-1], f1_scores[-1], f"{f1_scores[-1]}", color="red", fontsize=10, ha="left")

    # 设置图例、标题和轴标签
    plt.xlabel("Cosine Similarity Threshold", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title("Precision, Recall, and F1-Score vs Cosine Similarity Threshold", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{dim}_{topk}_cosine_threshold.png")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理蛋白质文件的脚本")
    parser.add_argument("--dim", type=int, required=True, help="faiss维度")
    parser.add_argument("--topk", type=int, required=True, help="faiss选取的个数")

    args = parser.parse_args()


    main(args.dim,args.topk)