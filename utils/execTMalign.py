import os
import subprocess
import re
import argparse
import pandas as pd

"""
AF-P85093-F1-model_v4
AF-P69115-F1-model_v4
AF-P0C5P5-F1-model_v4
AF-P85334-F1-model_v4
AF-P17536-F1-model_v4
AF-P85096-F1-model_v4

这六个文件使用foldseek搜索不到

"""
# 设置 TMalign 可执行文件的路径
tm_align_exec = "../bin/TMalign_cpp"


def extract_key_info(output):
    """
    提取 TMalign 输出中的关键信息。
    """
    key_info = ""

    # 使用正则表达式匹配并提取关键信息
    chain_1 = re.search(r"Name of Chain_1: .+", output)
    chain_2 = re.search(r"Name of Chain_2: .+", output)
    length_chain_1 = re.search(r"Length of Chain_1: \d+ residues", output)
    length_chain_2 = re.search(r"Length of Chain_2: \d+ residues", output)
    aligned_length = re.search(r"Aligned length= \d+, RMSD= +[\d.]+, Seq_ID=n_identical/n_aligned= [\d.]+", output)
    tm_score_1 = re.search(r"TM-score= [\d.]+ \(if normalized by length of Chain_1", output)
    tm_score_2 = re.search(r"TM-score= [\d.]+ \(if normalized by length of Chain_2", output)
    recommendation = re.search(r"\(You should use TM-score normalized by length of the reference structure\)", output)

    # 将匹配的内容拼接成字符串
    if chain_1:
        key_info += chain_1.group(0) + "\n"
    if chain_2:
        key_info += chain_2.group(0) + "\n"
    if length_chain_1:
        key_info += length_chain_1.group(0) + "\n"
    if length_chain_2:
        key_info += length_chain_2.group(0) + "\n"
    if aligned_length:
        key_info += aligned_length.group(0) + "\n"
    if tm_score_1:
        key_info += tm_score_1.group(0) + "\n"
    if tm_score_2:
        key_info += tm_score_2.group(0) + "\n"
    if recommendation:
        key_info += recommendation.group(0) + "\n"

    return key_info


def compare_with_target(directory, target_file, output_file):
    """
    与目标文件比对目录中的所有文件，并将关键信息写入输出文件。
    """

    # 获取目录中的所有结构文件
    structure_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".cif")]



    for file in structure_files:
        try:
            result = subprocess.run([tm_align_exec, target_file, file], capture_output=True, text=True, check=True)
            key_info = extract_key_info(result.stdout)
            with open(output_file, "a") as f:
                f.write(f"比对文件: {file} 和 {target_file}\n")
                f.write(key_info)
                f.write("---------------------------------------------\n")

        except subprocess.CalledProcessError as e:
            print(f"比对文件 {file} 和 {target_file} 时出现错误：{e}")
        except Exception as e:
            print(f'{e}')


def sort_Talign_chain(input_file_path,output_file_path):
    results = []

    # 逐行读取文件并解析内容
    with open(input_file_path, 'r') as f:
        # 一次性读取10行
        while True:
            # 读取每一组比对结果
            lines = [f.readline() for _ in range(10)]
            if not lines[0]:
                break  # 文件结束

            try:
                # 提取文件名
                file1 = re.search(r'比对文件: (.*?) 和', lines[0]).group(1)
                file2 = re.search(r' 和 (.*?)\n', lines[0]).group(1)

                # 提取对齐长度、RMSD、序列相似性
                aligned_length = int(re.search(r'Aligned length= (\d+)', lines[5]).group(1))
                rmsd = float(re.search(r'RMSD= +([\d.]+)', lines[5]).group(1))
                seq_id = float(re.search(r'Seq_ID=n_identical/n_aligned= ([\d.]+)', lines[5]).group(1))

                # 提取 TM-score（按 Chain_1 和 Chain_2 标准化）
                tm_score_1 = float(
                    re.search(r'TM-score= ([\d.]+) \(if normalized by length of Chain_1', lines[6]).group(1))
                tm_score_2 = float(
                    re.search(r'TM-score= ([\d.]+) \(if normalized by length of Chain_2', lines[7]).group(1))

                # 保存结果
                results.append({
                    "File1": file1,
                    "File2": file2,
                    "TM-Score1": tm_score_1,
                    "TM-Score2": tm_score_2,
                    "Aligned Length": aligned_length,
                    "RMSD": rmsd,
                    "Seq_ID": seq_id
                })
            except Exception as e:
                # 跳过解析失败的比对结果
                print(f"解析以下结果时出错，已跳过：{lines}\n错误：{e}")
    #
        # 转换为 DataFrame 并写入 CSV 文件
        df = pd.DataFrame(results)
        df.to_csv(output_file_path, index=False)



def main():
    """
    AF-P85093-F1-model_v4.cif
    AF-P69115-F1-model_v4.cif
    AF-P0C5P5-F1-model_v4.cif
    AF-P85334-F1-model_v4.cif
    AF-P17536-F1-model_v4.cif
    AF-P85096-F1-model_v4.cif
    """
    parser = argparse.ArgumentParser(description="tmalign处理的文件")
    parser.add_argument('--queryfile', type=str, required=True, help='输入文件名')
    args = parser.parse_args()



    structure_dir = "../data/pdb/Swissport"

    target_file = f"/data/foldseek_database/sp_cif_file/swissprot_cif_v4_files/{args.queryfile}.cif"

    output_file = f"/data/foldseek_database/tmalign/normal_file/tmalign_allsp_result/{args.queryfile}.result"

    try:
        compare_with_target(structure_dir, target_file, output_file)
    except Exception as e:
        print(f"程序中断时的异常信息：{e}")


if __name__ == "__main__":
    main()



