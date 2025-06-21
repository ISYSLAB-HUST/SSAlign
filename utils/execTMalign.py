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


"""
tm_align_exec = "../bin/TMalign_cpp"


def extract_key_info(output):
    
    key_info = ""

    # 
    chain_1 = re.search(r"Name of Chain_1: .+", output)
    chain_2 = re.search(r"Name of Chain_2: .+", output)
    length_chain_1 = re.search(r"Length of Chain_1: \d+ residues", output)
    length_chain_2 = re.search(r"Length of Chain_2: \d+ residues", output)
    aligned_length = re.search(r"Aligned length= \d+, RMSD= +[\d.]+, Seq_ID=n_identical/n_aligned= [\d.]+", output)
    tm_score_1 = re.search(r"TM-score= [\d.]+ \(if normalized by length of Chain_1", output)
    tm_score_2 = re.search(r"TM-score= [\d.]+ \(if normalized by length of Chain_2", output)
    recommendation = re.search(r"\(You should use TM-score normalized by length of the reference structure\)", output)

    # 
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

    
    with open(input_file_path, 'r') as f:
       
        while True:
            #
            lines = [f.readline() for _ in range(10)]
            if not lines[0]:
                break 

            try:
               
                file1 = re.search(r'比对文件: (.*?) 和', lines[0]).group(1)
                file2 = re.search(r' 和 (.*?)\n', lines[0]).group(1)

               
                aligned_length = int(re.search(r'Aligned length= (\d+)', lines[5]).group(1))
                rmsd = float(re.search(r'RMSD= +([\d.]+)', lines[5]).group(1))
                seq_id = float(re.search(r'Seq_ID=n_identical/n_aligned= ([\d.]+)', lines[5]).group(1))

                tm_score_1 = float(
                    re.search(r'TM-score= ([\d.]+) \(if normalized by length of Chain_1', lines[6]).group(1))
                tm_score_2 = float(
                    re.search(r'TM-score= ([\d.]+) \(if normalized by length of Chain_2', lines[7]).group(1))

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
                print(f"解析以下结果时出错，已跳过：{lines}\n错误：{e}")
    
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



