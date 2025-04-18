from Bio import SeqIO

"""
foldseek 创建的数据库文件如下，里面会有一些我们需要的信息，提取出来

chen@ubuntu:/data/foldseek_database/Scope40/Scope40DB/DB$ ls
scope40DB     scope40DB_ca.dbtype  scope40DB.dbtype  scope40DB_h.dbtype  scope40DB.index   scope40DB.source  scope40DB_ss.dbtype
scope40DB_ca  scope40DB_ca.index   scope40DB_h       scope40DB_h.index   scope40DB.lookup  scope40DB_ss      scope40DB_ss.index
(没有  _taxonomy 这个文件 )

chen@ubuntu:/data/foldseek_database/spDB$ ls
spDB             spDB_ca            spDB_h              spDB_ss
spDB.index       spDB_ca.index      spDB_h.index        spDB_ss.index
spDB.dbtype      spDB_ca.dbtype     spDB_h.dbtype       spDB_ss.dbtype

spDB.lookup
spDB_mapping
spDB_taxonomy


head spDB.dbtype  空的

head spDB
MSLEQKKGADIISKILQIQNSIGKTTSPSTLKTKLSEISRKEQENARIQSKLSDLQKKKIDIDNKLLKEKQNLIKEEILERKKLEVLTKKQQKDEIEHQKKLKREIDAIKASTQYITDVSISSYNNTIPETEPEYDLFISHASEDKEDFVRPLAETLQQLGVNVWYDEFTLKVGDSLRQKIDSGLRNSKYGTVVLSTDFIKKDWTNYELDGLVAREMNGHKMILPIWHKITKNDVLDYSPNLADKVALNTSVNSIEEIAHQLADVILNR
MTSHGAVKIAIFAVIALHSIFECLSKPQILQRTDHSTDSDWDPQMCPETCNPSKNISCSSECLCVTLGGGDETGTCFNMSGVDWLGHAQASDGHNDG
MKVLLYIAASCLMLLALNVSAENTQQEEEDYDYGTDTCPFPVLANKTNKAKFVGCHQKCNGGDQKLTDGTACYVVERKVWDRMTPMLWYSCPLGECKNGVCEDLRKKEECRKGNGEEK

------- 这个是什么序列？



head spDB.index
23805   173994608       381
23930   227306  595
37512   186172557       134


head spDB.lookup
23805   AF-Q94WX5-F1-model_v4   0
23930   AF-A0A0B5EMG9-F1-model_v4       0
37512   AF-Q9RPW4-F1-model_v4   0


head spDB_mapping
23805   61874
23930   5507
37512   39775




"""


"""
将 spDB.lookup 中的文件名提取出来，给下一步使用

然后执行：
foldseek createsubdb accession_list alphafold_swissport afsp_subset --id-mode 1
foldseek convert2fasta afsp_subset afsp_subset.fasta

获取到  afsp_subset.fasta 就是 残基序列的 fasta文件

ln -s spDB.lookup spDB_ss.lookup   # 因为本身是没有 spDB_ss.lookup 的，所以建立软连接，使用spDB.lookup
foldseek createsubdb accession_list alphafold_swissport_ss afsp_subset_ss --id-mode 1
foldseek lndb alphafold_swissport_h afsp_subset_ss_h
foldseek convert2fasta afsp_subset_ss afsp_subset_ss.fasta

获取到 afsp_subset_ss.fasta  就是3Di序列的 fasta文件



"""
def make_fasta():
    with open('/data2/zxc_data/afdb50/afdb50.lookup', 'r') as infile:
        # 打开新文件进行写入
        with open('/data2/zxc_data/afdb50/accession_list', 'w') as outfile:
            # 遍历原文件的每一行
            for line in infile:
                # 按照制表符或空格分割每一行
                columns = line.split()
                if len(columns) > 1:  # 确保至少有两列
                    # 提取第二列并写入新文件
                    outfile.write(columns[1] + '\n')


def integrate_sequences(residue_seq, ss_seq):
    integrated_seq = []
    for res, ss in zip(residue_seq, ss_seq):
        integrated_seq.append(res + ss.lower())  # 大写残基 + 小写 3Di
    return "".join(integrated_seq)

def combine_fasta(residue_file, ss_file, output_file):
    # 打开输出文件
    with open(output_file, "w") as outfile:
        # 读取残基序列文件和 3Di 序列文件
        residue_records = SeqIO.parse(residue_file, "fasta")
        ss_records = SeqIO.parse(ss_file, "fasta")

        for residue_record, ss_record in zip(residue_records, ss_records):
            # 检查 ID 是否匹配
            if residue_record.id != ss_record.id:
                print(f"Warning: ID mismatch: {residue_record.id} vs {ss_record.id}")
                continue

            # 整合两条序列
            combined_sequence = integrate_sequences(str(residue_record.seq), str(ss_record.seq))

            # 写入新的 FASTA 文件
            outfile.write(f">{residue_record.id} {residue_record.description}\n")
            outfile.write(f"{combined_sequence}\n")

"""

    afsp_subset.fasta
>AF-Q94WX5-F1-model_v4 Cytochrome b
MTNTRKSHPLIKIVNHSFIDLPAPSNISAWWNFGSLLGVCLGLQILTGLFLAMHYTADTTTAFSSVTHICRDVNYGWLIRYMHANGASMFFIFLYFHIGRGIYYGSYTFMDTWNIGVLLLFAVMATAFMGYVLPWGQMSFWGATVITNLLSAIPYIGPTLVEWIWGGFSVDKATLTRFFAFHFILPFIITAMVMIHLLFLHETGSNNPSGMNSDSDKIPFHPYYTIKDILGILFMMITLMSLVMFTPDLLGDPDNYTPANPLNTPPHIKPEWYFLFAYAILRSIPNKLGGVLALVFSILILMLFPILHSSKQRSMSFRPLSQCLMWMLVANLLILTWIGGQPVEHPFITIGQLASVTYFFTILILMPSTALMENKLLKW
>AF-A0A0B5EMG9-F1-model_v4 Efflux pump FUBT
MAIDPQPSSPSLSSETIANDTIGNDNNVNEPSVEPKTQENQHTVPPRLSRIYSQAHHISQSFIDQNYPGEGTTQAPYRINFLPDDTQNAQLLPRWKKWAFVLLQSLACLATTFASSAYSGGIKQVIRAFGISQEVATLGISLYVLGFTFGPLIWAPLSELYGRKKVFFFTFMVATAFSAGAAGAGSIASLLVLRFLTGSIGSAPLSNAPALIADMFDKSERGLAMCMFSGAPFLGPAIGPIAGGFLGEAAGWRWLHGLMAAFTGVTWIACTVFIPETYAPYILRKRAQHMSKLTGKVYISTLDADKPPSSAAHQLKNALTRPWLLLFKEPIVFITSIYISIIYGTMYMCFAAFPIVFQKGRGWSQGIGGLAFTGIVIGVVLSIISFAFEDKRYARAAQRRGAPMEPEDRLPPAIMGSLLIPIGLFWFAWTTFPSVHWIVPIIGTVFFAWGLVLVFMALLNYLIDSYVIFAASIMAANSALRSLFGAAFPLFTRQMYDGLGVQWASSIPAFLALACVPFPFLFYKYGRQIRMKCEYAAEAANVLQKMRSLHVTVTEDDAMNEAEEMWRARTHNSHASAAHSHGHRRSLSYTRSV


    afsp_subset_ss.fasta
>AF-Q94WX5-F1-model_v4 Cytochrome b
DPDCCCPPPVSVVVCCQAFQPWAFPFADPLVLLLVVLVVLLVLLLVLVVVLVVQADLALVCQLVSLVCLCPPDDCSLVSVLLNVVSLVVNLVSLVSNLLVCLQLLVCLLVQLLVLSLVLVVLSVLLVLLVLLRNQFQLNFVVVQVVLLLQLQPPPCSCVSSCVCCVHNGNHSSNSVVSVVSSNCSSVVSVVSSVSSSVSCSVPDDAHQQRDDRPVDIDTNPPPVVVVSVVVVVVVVVVSVCCSPPPSCPQDDPRRNHHHDPVDHDPDRADGLSLQLLLLQLLLDPGSVVSVVLSVCLSVLSSCSSVLDQAPGSHCPVPVVLVVLSVVLVVLSVQSSVLSRDDPDPPSNVSSNVSSVSNCCSSNPVNSVSRVVVCVVVVD
>AF-A0A0B5EMG9-F1-model_v4 Efflux pump FUBT
DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDPPPPPPPPPPDDPVVCLVCVLVDDDPCNLPDDFFFDLAPVTRGADDADPPRPLQLQPDDPVLLVVLLVLLLLLLLLLLLLLALLLLQLVVVCVVAVDDSLLSLLLNLLLLLLLLQQLLPLVLVCLVVFVLVLLLVLLVLLLVLLLVLLVDPHSVSNSVSSNSSSNSSCNSVNVSLVLLVLNDDPVPSLVSCLVSLLSNQVSLLCSLQNSNVCCQVPRSSVSSVVSSVSSVVSSVVCSVSPGGRNRQVSQQSVQVSSCVVVVGHYDYPVCPPPDDDDSVRVSVCSVPLLVVCCVQAPLLVLLLVLLQLLLLVSLLCSQLLCVLCCVVVVDGSNVSSCLSVLLVLLSVLLSVVVVVLVVVLVVVCVVVVDDDALLSLLPLLLQLLVLQLQLLLQLLVQSDPPHPPVRNSPSSSSNSNSSSSSNVSSLSSLCRLQVQCSSSSVSSSSNSNSNSSSPSNSCLVVLCVVVNRNVSSCPSSVVSNVSNCPSVCCSVCVVVSLCVRQNSVLSVVVVVVCVVVVDDDHSVNSVVVSVVVVVVVVVVVVVVVVVVVVVVVVVVVVPDD



    afsp_combined.fasta
>AF-Q94WX5-F1-model_v4 AF-Q94WX5-F1-model_v4 Cytochrome b
MdTpNdTcRcKcSpHpPpLvIsKvIvVvNcHcSqFaIfDqLpPwAaPfSpNfIaSdApWlWvNlFlGlSvLvLlGvVvClLlGvLlQlIlLvTlGvLvFvLlAvMvHqYaTdAlDaTlTvTcAqFlSvSsVlTvHcIlCcRpDpVdNdYcGsWlLvIsRvYlMlHnAvNvGsAlSvMvFnFlIvFsLlYvFsHnIlGlRvGcIlYqYlGlSvYcTlFlMvDqTlWlNvIlGsVlLvLlLvFvAlVsMvAlTlAvFlMlGvYlVlLrPnWqGfQqMlSnFfWvGvAvTqVvIvTlNlLlLqSlAqIpPpYpIcGsPcTvLsVsEcWvIcWcGvGhFnSgVnDhKsAsTnLsTvRvFsFvAvFsHsFnIcLsPsFvIvIsTvAvMsVsMvIsHsLsLvFsLcHsEvTpGdSdNaNhPqSqGrMdNdSrDpSvDdKiIdPtFnHpPpYpYvTvIvKvDsIvLvGvIvLvFvMvMvIvTvLsMvScLcVsMpFpTpPsDcLpLqGdDdPpDrNrYnThPhAhNdPpLvNdThPdPpHdIrKaPdEgWlYsFlLqFlAlYlAlIqLlRlSlIdPpNgKsLvGvGsVvLvAlLsVvFcSlIsLvIlLsMsLcFsPsIvLlHdSqSaKpQgRsShMcSpFvRpPvLvSlQvCvLlMsWvMvLlVvAvNlLsLvIqLsTsWvIlGsGrQdPdVpEdHpPpFsInTvIsGsQnLvAsSsVvTsYnFcFcTsIsLnIpLvMnPsSvTsArLvMvEvNcKvLvLvKvWd
>AF-A0A0B5EMG9-F1-model_v4 AF-A0A0B5EMG9-F1-model_v4 Efflux pump FUBT
MdAdIdDdPdQdPdSdSdPdSdLdSdSdEdTdIdAdNdDdTdIdGdNdDdNdNdVdNdEdPdSdVdEdPpKpTpQpEpNpQpHpTpVpPdPdRpLvSvRcIlYvScQvAlHvHdIdSdQpScFnIlDpQdNdYfPfGfEdGlTaTpQvAtPrYgRaIdNdFaLdPpDpDrTpQlNqAlQqLpLdPdRpWvKlKlWvAvFlVlLvLlQlSlLlAlClLlAlTlTlFlAlSlSaAlYlSlGlGqIlKvQvVvIcRvAvFaGvIdSdQsElVlAsTlLlGlInSlLlYlVlLlGlFlTlFqGqPlLlIpWlAvPlLvScElLvYvGfRvKlKvVlFlFlFvTlFlMvVlAlTlAvFlSlAlGvAlAlGvAdGpShIsAvSsLnLsVvLsRsFnLsTsGsSnIsGsScAnPsLvSnNvAsPlAvLlIlAvDlMnFdDdKpSvEpRsGlLvAsMcClMvFsSlGlAsPnFqLvGsPlAlIcGsPlIqAnGsGnFvLcGcEqAvApGrWsRsWvLsHsGvLvMsAsAvFsTsGvVvTsWsIvAvCcTsVvFsIpPgEgTrYnArPqYvIsLqRqKsRvAqQvHsMsScKvLvTvGvKgVhYyIdSyTpLvDcApDpKpPdPdSdSdAsAvHrQvLsKvNcAsLvTpRlPlWvLvLcLcFvKqEaPpIlVlFvIlTlSlIvYlIlSqIlIlYlGlTvMsYlMlCcFsAqAlFlPcIvVlFcQcKvGvRvGvWdSgQsGnIvGsGsLcAlFsTvGlIlVvIlGlVsVvLlSlIsIvSvFvAvFvElDvKvRvYlAvRvAvAcQvRvRvGvAdPdMdEaPlElDsRlLlPpPlAlIlMqGlSlLvLlIqPlIqGlLlFlWqFlAlWvTqTsFdPpSpVhHpWpIvVrPnIsIpGsTsVsFsFnAsWnGsLsVsLsVsFnMvAsLsLlNsYsLlIcDrSlYqVvIqFcAsAsSsIsMvAsAsNsSsAnLsRnSsLnFsGsAsApFsPnLsFcTlRvQvMlYcDvGvLvGnVrQnWvAsSsScIpPsAsFvLvAsLnAvCsVnPcFpPsFvLcFcYsKvYcGvRvQvIsRlMcKvCrEqYnAsAvElAsAvNvVvLvQvKvMcRvSvLvHvVdTdVdThEsDvDnAsMvNvEvAsEvEvMvWvRvAvRvTvHvNvSvHvAvSvAvAvHvSvHvGvHvRvRvSvLvSvYvTvRpSdVd



"""


if __name__ == "__main__":
    # make_fasta()

    # 使用文件
    # combine_fasta("/data/foldseek_database/spDB/afsp_subset.fasta", "/data/foldseek_database/spDB/afsp_subset_ss.fasta", "/data/foldseek_database/spDB/afsp_combined.fasta")
    
    # make_fasta()
    combine_fasta("/data2/zxc_data/afdb50/afdb50_subset.fasta","/data2/zxc_data/afdb50/afdb50_subset_ss.fasta","/data2/zxc_data/afdb50/afdb50_combined.fasta")

