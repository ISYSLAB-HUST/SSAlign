首先使用foldseek将 afdb50 数据库 重建为fasta文件，并且计算所有的蛋白质的嵌入

参考方法：

`foldseek createsubdb accession_list alphafold_swissport afsp_subset --id-mode 1`

`foldseek convert2fasta afsp_subset afsp_subset.fasta`

获取到  afsp_subset.fasta 就是 残基序列的 fasta文件

`ln -s spDB.lookup spDB_ss.lookup`   # 因为本身是没有 spDB_ss.lookup 的，所以建立软连接，使用spDB.lookup

`foldseek createsubdb accession_list alphafold_swissport_ss afsp_subset_ss --id-mode 1`

`foldseek lndb alphafold_swissport_h afsp_subset_ss_h`

`foldseek convert2fasta afsp_subset_ss afsp_subset_ss.fasta`

