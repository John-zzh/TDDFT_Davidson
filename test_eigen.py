import numpy as np
import eigen_solver
import time

size = 5000
A = np.random.rand(size, size) 
A = (A + A.T)/10
A = np.dot(A,A.T) #正定矩阵
np.fill_diagonal(A, np.diag(A)*10)  # 增加对角线上的元素
diag = np.diag(A)



def gen_matrix_vector_product(A):
    def matrix_vector_product_callable(V):
        return np.dot(A,V)
    return matrix_vector_product_callable

matrix_vector_product_callable = gen_matrix_vector_product(A)

start = time.time()
energies, X = eigen_solver.Davidson(matrix_vector_product_callable, diag,
                                    N_states = 5,
                                    conv_tol = 1e-5,
                                    max_iter = 50)
end = time.time()
print("Davidson time: ", end - start)
print(energies)
print()

print('============ full diagonalization with numpy.linalg.eig ====================')
print('running...')
start = time.time()
energies, X = np.linalg.eig(A)
# 对特征值从小到大排序
idx = np.argsort(energies)  # 获取排序的索引
energies = energies[idx]  # 按索引排序特征值
X = X[:, idx]  # 按索引排序特征向量
end = time.time()
print("numpy.linalg.eig time: ", end - start)
print(energies[:5])



