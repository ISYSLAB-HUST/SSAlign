import csv





if __name__=="__main__":

    input_list_file = "/data2/zxc_data/foldseek_database/foldseek_database/sp_cif_file/gitdata/SSAlign/SVD1280/100filenames.txt"
   
    dim = 512
    cosine_threshold = 0.3  # 相似度阈

    dim = 256
    cosine_threshold = 0.45  # 相似度阈

    dim = 128
    cosine_threshold = 0.6  # 相似度阈

    dim = 64
    cosine_threshold = 0.7  # 相似度阈
    
    # 配置参数
    output_file = f'svd{dim}_cos_greater_{cosine_threshold}.csv'  # 最终输出文件

    # 读取文件名列表
    with open(input_list_file, 'r') as f:
        csv_files = [f"/data2/zxc_data/foldseek_database/foldseek_database/sp_cif_file/gitdata/SSAlign/SVD{dim}/{line.strip()}.result" for line in f if line.strip()]
    
   

    # 处理所有CSV文件并合并结果
    with open(output_file, 'w', newline='') as outfile:
        writer = None  # CSV writer对象稍后初始化
        files_processed = 0

        for csv_file in csv_files:
            try:
                with open(csv_file, 'r') as infile:
                    reader = csv.DictReader(infile)
    
                    # 如果是第一个文件，初始化writer
                    if writer is None:
                        fieldnames = ['File1', 'File2', 'Cosine_Similarity', 'Avg_TM_Score']
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
    
                    # 处理当前文件
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


            except FileNotFoundError:
                print(f"警告：文件 {csv_file} 未找到，已跳过")
            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
        
    print(f"\n处理完成！已处理 {files_processed}/{len(csv_files)} 个文件")
    print(f"结果已保存到 {output_file}")
