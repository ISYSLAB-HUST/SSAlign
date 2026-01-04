#!/bin/bash

# 输入文件列表
input_file="./filenames_without_extension.txt"

# 日志文件
log_file="./logs/foldseek_processing_times.log"

# 清空或创建日志文件
> "$log_file"

# 逐行读取文件名并逐个处理
while IFS= read -r filename; do
    # 构造输入文件路径
    input_cif="../pdbData/pdb/AFDB50/${filename}.cif"
    
    # 构造输出文件路径
    output_foldseek="${filename}.foldseek"
    
    echo "Processing file: $filename"
    
    # 执行命令并记录时间
    time_output=$( { /usr/bin/time -f "%e" ../bin/foldseek easy-search "$input_cif" ../models/foldseekDB/afdb50/afdb50 "$output_foldseek" ../benchmarkData/AFDB50/foldseek/timebenchmark/ --threads 64; } 2>&1 )
    
    # 提取耗时（以秒为单位）
    elapsed_time=$(echo "$time_output" | tail -n 1)
    
    # 将结果写入日志文件
    echo "$filename: $elapsed_time seconds" >> "$log_file"
    
    # 检查命令是否成功
    if [ $? -eq 0 ]; then
        echo "Finished processing: $filename (Time: ${elapsed_time}s)"
    else
        echo "Error processing: $filename"
    fi
done < "$input_file"

echo "All files processed. Check the log file: $log_file"

