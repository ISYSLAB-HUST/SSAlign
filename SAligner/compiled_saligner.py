from numba.pycc import CC

# 创建 pycc 编译实例，指定输出文件名
cc = CC('saligner')  # 编译后的共享库文件名

cc.verbose = True  # 打开详细的编译信息

# 从 test.py 导入函数并注册到编译模块
from pair_align import saligner

# 注册函数及其签名
cc.export('saligner', 'i8(string, string)')(saligner)

if __name__ == '__main__':
    cc.compile()
