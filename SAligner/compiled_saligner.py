from numba.pycc import CC

cc = CC('saligner') 

cc.verbose = True  

from pair_align import saligner

cc.export('saligner', 'i8(string, string)')(saligner)

if __name__ == '__main__':
    cc.compile()
