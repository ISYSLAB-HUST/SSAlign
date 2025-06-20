import csv
import glob
import os


dim = 512
cosine_threshold = 0.3  # 

dim = 256
cosine_threshold = 0.45  # 

dim = 128
cosine_threshold = 0.6  # 

dim = 64
cosine_threshold = 0.7  # 

# 
input_dir = f"../gitdata/SSAlign/SVD{dim}"
output_file = f"scope40_svd{dim}_cos_greater_{cosine_threshold}.csv"  # 

csv_files = glob.glob(os.path.join(input_dir, "*.result"))

with open(output_file, 'w', newline='') as outfile:
    writer = None  # 
    files_processed = 0
    
    for csv_file in csv_files:
        try:
            with open(csv_file, 'r') as infile:
                reader = csv.DictReader(infile)
                
                if writer is None:
                    fieldnames = ['File1', 'File2', 'Cosine_Similarity', 'Avg_TM_Score']
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()
                
                # 
                rows_processed = 0
                for row in reader:
                    try:
                        cosine = float(row['Cosine_Similarity'])
                        if cosine > cosine_threshold:
                            avg_tm = (float(row['TM-Score1']) + float(row['TM-Score2'])) / 2
                            writer.writerow({
                                'File1': row['File1'],
                                'File2': row['File2'],
                                'Cosine_Similarity': cosine,
                                'Avg_TM_Score': avg_tm
                            })
                            rows_processed += 1
                    except (ValueError, KeyError) as e:
                        print(f"在文件 {csv_file} 中跳过格式错误的行: {e}")
                        continue
                
                files_processed += 1
               # print(f"已处理 {csv_file}，找到 {rows_processed} 条符合条件的记录")
                
        except FileNotFoundError:
            print(f"警告：文件 {csv_file} 未找到，已跳过")
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")

