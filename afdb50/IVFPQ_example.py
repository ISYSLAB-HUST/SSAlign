import faiss
import numpy as np

if __name__=="__main__":

    dim = 128  
    nlist = 100  
    m = 16 
    nbits = 8  
    n = 10000  


    xb = np.random.rand(n, dim).astype('float32')

    quantizer = faiss.IndexFlatIP(dim)  

    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    index.train(xb)  
    index.add(xb)  

    k = 5  
    xq = np.random.rand(5, dim).astype('float32')

    faiss.normalize_L2(xq)

    distances, indices = index.search(xq, k)

    print("Distances (Inner Product):\n", distances)
    print("Indices:\n", indices)

