import matplotlib as mpl 
import numpy as np 
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
mpl.use("agg")

mpl.rcParams["figure.dpi"] = 300

n_features = 40
group_size = 10
mean_list = [3, 4, 8]
var_list = [1,2,3]
labels = []
obs_list = []

for mu, sigma in zip(mean_list, var_list):
    x = mu + sigma * np.random.randn(group_size, n_features)
    obs_list.append(x)
    labels += [mu]*group_size

A = np.concatenate(obs_list, axis=0)
n_rows = A.shape[0]
C_A = cosine_similarity(A)

B = A.copy()
for row in B:
    row /= np.linalg.norm(row)

# Up to this point the matrix B has been normalized.

C_B = B @ B.T

delta_similarity = np.linalg.norm(C_A-C_B,1)
print(f"{delta_similarity=}")

d_vec = C_B @ np.ones(n_rows)
D = sp.diags(d_vec)
L = D - C_B
obj = eigsh(L, k=2, M=D, sigma=0, which="LM")
eigen_val_abs = np.abs(obj[0])
idx = np.argmax(eigen_val_abs)
eigen_vecs = obj[1]
partition = 0 < eigen_vecs[:,idx]

# Random partition
p = np.array([3,4,6,7,23])
n_p = len(p)
L_partition = L[p,:]
L_partition = L_partition[:,p]
B_partition = B[p]
C_partition = B_partition @ B_partition.T
D_partition = sp.diags(C_partition @ np.ones(n_p))
L_partition_clone = D_partition - C_partition
delta_matrix = np.linalg.norm(
    L_partition - L_partition_clone, 1)
print(f"{delta_matrix=}")





# Fast algorithm
d_inv_sqrt = 1/np.sqrt(d_vec)
D_inv_sqrt = sp.diags(d_inv_sqrt)
C = D_inv_sqrt @ B
trunc_SVD = TruncatedSVD(
    n_components=2,
    n_iter=5,
    algorithm='randomized')
W = trunc_SVD.fit_transform(C)
fast_partition = 0 < W[:,1]

if fast_partition[0] != partition[0]:
    fast_partition = ~fast_partition

fast_partition = fast_partition.astype(float)
partition = partition.astype(float)
delta_partition = np.linalg.norm(partition - fast_partition)
print(f"{delta_partition=}")




exit()




min_value = A.min()
A += -min_value

C2 = cosine_similarity(A)

delta = np.linalg.norm(C1-C2,1)
print(f"{delta=}")

exit()



A = AnnData(A)
A.obs["batch"] = labels
sc.tl.pca(A, 
          n_comps=2, 
          zero_center = True, 
          svd_solver="randomized")
sc.pl.pca(A, color="batch", show=False)
plt.savefig("pca.png", bbox_inches="tight")
print(A)

ones = np.ones((3,3))
out = np.linalg.svd(ones)
print(out)



