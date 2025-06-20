import faiss
import numpy as np

if __name__=="__main__":
    dim = 128  
    n = 10000 

    xb = np.random.rand(n, dim).astype('float32')


    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(xb)
    index.add(xb)  

    k = 5 
    xq = np.random.rand(5, dim).astype('float32')

    faiss.normalize_L2(xq)

    distances, indices = index.search(xq, k)

    print("Distances (Inner Product):\n", distances)
    print("Indices:\n", indices)

