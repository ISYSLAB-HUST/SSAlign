"""
3Di序列全局比对 各种方法的时间测试
只计算比对得分即可
"""
import random
import time
import numpy as np
from Bio.Align import PairwiseAligner,substitution_matrices
from saligner import saligner
from pair_align import needleman_wunsch_gotoh_score_opm


"""
方法一：直接使用biopython库计算



"""
def use_bio(target, query):
    # 初始化全局比对器
    aligner = PairwiseAligner()

    # 设置参数     Needleman-Wunsch
    aligner.mode = "global"  # 全局比对

    # 使用3Di替换矩阵 mat3di.out
    # aligner.substitution_matrix = substitution_matrices.read("../models/mat3di.out")  # 加载替代矩阵文件
    aligner.substitution_matrix = substitution_matrices.read("/home/xuchaozhang/ssalign/ssalign/SaprotProject/Saprot/SaProt/Linearfitting/images/bioscore/3Di/mat3di.out")  # 加载替代矩阵文件

    # 设置罚分之后。 global 中的 SW算法 就变成了  Gotoh 算法 Gotoh global alignment algorithm
    aligner.open_gap_score = -10 # gap 开罚
    aligner.extend_gap_score = -1  # gap 延伸罚

    print(aligner.algorithm)

    score = aligner.score(target,query)
    print(f"biopython 得分 : {score}")

"""
使用 affine_gaps 来计算
其中使用了python的 numba编译器(JIT即时编译器)，可以加快速度（第一次调用需要编译，后就面不需要了）
"""
def use_saligner(target, query):

    score = saligner(query, target)
    print("saligner 得分 :", score)

def use_pairalign(target, query):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    substitutions = np.array([
        [6  , -3 , 1, 2, 3, -2, -2, -7, -3, -3, -10, -5, -1, 1, -4, -7, -5, -6, 0, -2, 0],
        [-3 , 6  , -2, -8, -5, -4, -4, -12, -13, 1, -14, 0, 0, 1, -1, 0, -8, 1, -7, -9, 0],
        [1  , -2 , 4, -3, 0, 1, 1, -3, -5, -4, -5, -2, 1, -1, -1, -4, -2, -3, -2, -2, 0],
        [2  , -8 , -3, 9, -2, -7, -4, -12, -10, -7, -17, -8, -6, -3, -8, -10, -10, -13, -6, -3, 0],
        [3  , -5 , 0, -2, 7, -3, -3, -5, 1, -3, -9, -5, -2, 2, -5, -8, -3, -7, 4, -4, 0],
        [-2 , -4 , 1, -7, -3, 6, 3, 0, -7, -7, -1, -2, -2, -4, 3, -3, 4, -6, -4, -2, 0],
        [-2 , -4 , 1, -4, -3, 3, 6, -4, -7, -6, -6, 0, -1, -3, 1, -3, -1, -5, -5, 3, 0],
        [-7 , -12, -3, -12, -5, 0, -4, 8, -5, -11, 7, -7, -6, -6, -3, -9, 6, -12, -5, -8, 0],
        [-3 , -13, -5, -10, 1, -7, -7, -5, 9, -11, -8, -12, -6, -5, -9, -14, -5, -15, 5, -8, 0],
        [-3 , 1, -4, -7, -3, -7, -6, -11, -11, 6, -16, -3, -2, 2, -4, -4, -9, 0, -8, -9, 0],
        [10 , -14, -5, -17, -9, -1, -6, 7, -8, -16, 10, -9, -9, -10, -5, -10, 3, -16, -6, -9, 0],
        [-5 , 0, -2, -8, -5, -2, 0, -7, -12, -3, -9, 7, 0, -2, 2, 3, -4, 0, -8, -5, 0],
        [-1 , 0, 1, -6, -2, -2, -1, -6, -6, -2, -9, 0, 4, 0, 0, -2, -4, 0, -4, -5, 0],
        [1  , 1, -1, -3, 2, -4, -3, -6, -5, 2, -10, -2, 0, 5, -2, -4, -5, -1, -2, -5, 0],
        [-4 , -1, -1, -8, -5, 3, 1, -3, -9, -4, -5, 2, 0, -2, 6, 2, 0, -1, -6, -3, 0],
        [-7 , 0, -4, -10, -8, -3, -3, -9, -14, -4, -10, 3, -2, -4, 2, 6, -6, 0, -11, -9, 0],
        [-5 , -8, -2, -10, -3, 4, -1, 6, -5, -9, 3, -4, -4, -5, 0, -6, 8, -9, -5, -5, 0],
        [-6 , 1, -3, -13, -7, -6, -5, -12, -15, 0, -16, 0, 0, -1, -1, 0, -9, 3, -10, -11, 0],
        [0  , -7, -2, -6, 4, -4, -5, -5, 5, -8, -6, -8, -4, -2, -6, -11, -5, -10, 8, -6, 0],
        [-2 , -9, -2, -3, -4, -2, 3, -8, -8, -9, -9, -5, -5, -5, -3, -9, -5, -11, -6, 9, 0],
        [0  , 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype=np.int8)

    score = needleman_wunsch_gotoh_score_opm(
        target, query,
        substitution_alphabet=alphabet,
        substitution_matrix=substitutions,
        gap_opening=-10,
        gap_extension=-1)

    print("pair_align 得分 :", score)





if __name__=="__main__":
    letters = 'ACDEFGHIKLMNPQRSTVWYX'

    # 生成长度为 1000 的随机序列
    s1 = ''.join(random.choices(letters, k=1000))

    s2 = ''.join(random.choices(letters, k=1000))

    print(s1)
    print(s2)


    time0 = time.time()

    use_saligner(s1, s2)

    time1 = time.time()

    use_bio(s1,s2)

    time2 = time.time()

    use_pairalign(s1, s2)

    time3 = time.time()

    use_pairalign(s1,s2)

    time4 = time.time()

    print(f"biopython用时：{time2-time1}")
    print(f"saligner用时：{time1-time0}")
    print(f"pairalign第一次用时：{time3-time2}")
    print(f"pairalign第二次用时：{time4-time3}")

"""

saligner 得分 : -216
Gotoh global alignment algorithm
biopython 得分 : -266.0
pair_align 得分 : -199
pair_align 得分 : -199
biopython用时：0.007706165313720703
saligner用时：0.003971099853515625
pairalign第一次用时：3.3678314685821533
pairalign第二次用时：0.0043179988861083984

"""