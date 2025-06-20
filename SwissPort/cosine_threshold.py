import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse


"""

"""


def cos_test(basenames, dim,topk,cos_threshold):
    combined_data = []

    for basename in basenames:
        csv_file_path = f"../data/result/Swissport/SSAlign/SVD{dim}/{basename}.result"

        df = pd.read_csv(csv_file_path)

        df_sorted = df.sort_values(by='Cosine_Similarity', ascending=False).head(topk)

        cosine_similarity = df_sorted['Cosine_Similarity']
        tm_score1 = df_sorted['TM-Score1']
        tm_score2 = df_sorted['TM-Score2']

        avg_tmscore = (tm_score1 + tm_score2) / 2
        combined_score = avg_tmscore


        combined_data.append({
            'Cosine_Similarity': cosine_similarity,
            'Avg_TM_Score': avg_tmscore,
            'combined_score': combined_score
        })

    combined_df = pd.concat([pd.DataFrame(data) for data in combined_data], ignore_index=True)

    filtered_df = combined_df[combined_df['Cosine_Similarity'] < cos_threshold]

    count = filtered_df.shape[0]


    filtered_1 = combined_df[(combined_df['Cosine_Similarity'] < cos_threshold) & (combined_df['Avg_TM_Score'] > 0.5)]
    count_1 = len(filtered_1)

    filtered_2 = combined_df[(combined_df['Cosine_Similarity'] < cos_threshold) & (combined_df['Avg_TM_Score'] < 0.5)]
    count_2 = len(filtered_2)
    #print(f"Cosine_Similarity < {cos_threshold} and Avg_TM_Score < 0.5 的数据量: {count_2}")

    filtered_3 = combined_df[(combined_df['Cosine_Similarity'] > cos_threshold) & (combined_df['Avg_TM_Score'] > 0.5)]
    count_3 = len(filtered_3)
    #print(f"Cosine_Similarity >  {cos_threshold} and Avg_TM_Score > 0.5 的数据量: {count_3}")

    filtered_4 = combined_df[(combined_df['Cosine_Similarity'] >  cos_threshold) & (combined_df['Avg_TM_Score'] < 0.5)]
    count_4 = len(filtered_4)
    #print(f"Cosine_Similarity >  {cos_threshold} and Avg_TM_Score < 0.5 的数据量:  {count_4}")

    # 
    precision = count_3 / (count_3 + count_4) if count_3 + count_4 > 0 else 0

    # 
    recall = count_3 / (count_3 + count_1) if count_3 + count_1 > 0 else 0

    # 
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score


def main(dim,topk):
    with open("100filenames.txt", "r") as f:
        basenames = [line.strip() for line in f if line.strip()]  # 

    # cos_test(basenames, SVD1280,1000,0.2)

    thresholds = np.arange(0.1, 1.0, 0.001)  # 
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        precision, recall, f1_score = cos_test(basenames, dim,topk,threshold)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    plt.plot(thresholds, precisions, label="Precision", color="blue", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall", color="green", linewidth=2)
    plt.plot(thresholds, f1_scores, label="F1-Score", color="red", linewidth=2)

    # 
    plt.text(thresholds[0], precisions[0], f"{precisions[0]}", color="blue", fontsize=10, ha="right")
    plt.text(thresholds[-1], precisions[-1], f"{precisions[-1]}", color="blue", fontsize=10, ha="left")
    plt.text(thresholds[0], recalls[0], f"{recalls[0]}", color="green", fontsize=10, ha="right")
    plt.text(thresholds[-1], recalls[-1], f"{recalls[-1]}", color="green", fontsize=10, ha="left")
    plt.text(thresholds[0], f1_scores[0], f"{f1_scores[0]}", color="red", fontsize=10, ha="right")
    plt.text(thresholds[-1], f1_scores[-1], f"{f1_scores[-1]}", color="red", fontsize=10, ha="left")

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
