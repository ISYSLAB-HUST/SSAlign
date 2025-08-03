First, use foldseek to rebuild the afdb50 database into a fasta file and calculate the embedding of all proteins.

Reference method:

`foldseek createsubdb accession_list alphafold_swissport afsp_subset --id-mode 1`

`foldseek convert2fasta afsp_subset afsp_subset.fasta`

Obtain  afsp_subset.fasta, which is the fasta file of the residue sequence.

`ln -s spDB.lookup spDB_ss.lookup`   

`foldseek createsubdb accession_list alphafold_swissport_ss afsp_subset_ss --id-mode 1`

`foldseek lndb alphafold_swissport_h afsp_subset_ss_h`

`foldseek convert2fasta afsp_subset_ss afsp_subset_ss.fasta`

