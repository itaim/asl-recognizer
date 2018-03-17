from  my_data_preparation import *
from numpy.lib.stride_tricks import as_strided as strided
import matplotlib as plt


def get_sliding_window(df, W, return2D=0):
    a = df.values
    s0,s1 = a.strides
    m,n = a.shape
    out = strided(a,shape=(m-W+1,W,n),strides=(s0,s0,s1))
    if return2D==1:
        return out.reshape(a.shape[0]-W+1,-1)
    else:
        return out

# A Summary of the PCA Approach
# Standardize the data.
# Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.

# Sort eigenvalues in descending order and choose the k eigenvectors that correspond to the k largest eigenvalues where k is the number of dimensions of the new feature subspace (k≤d).
# Construct the projection matrix W from the selected k eigenvectors.
# Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.


d = 4
k = 3

def ev(arr):
    print("input shape {}".format(arr.shape))
    eig_val,eig_vec = np.linalg.eig((np.outer(arr, np.transpose(arr))))
    indices = [i for (i,v) in sorted(enumerate(eig_val),key = lambda x:x[1],reverse = True)][0:k]
    eig_vec = np.asarray([ev for (i,ev) in enumerate(eig_vec) if i in indices])

    x = arr[0]
    # print, evec(type(arr))
    print("arr \n{}".format(np.array2string(eig_vec)))
    # print("eval \n{}".format(np.array2string(eig_val)))
    # print("evec \n{}".format(np.array2string(eig_vec)))
    # print("foo")
    return sum(arr)


def trajectory(a):
    arr = a.reshape(5,4)
    print("input shape {}".format(arr.shape))
    print(np.array2string(arr))
    eig_val, eig_vec = np.linalg.eig((np.outer(arr, np.transpose(arr))))
    indices = [i for (i, v) in sorted(enumerate(eig_val), key=lambda x: x[1], reverse=True)][0:eig_vec.shape[0]//4*3]
    eig_vec = np.asarray([ev for (i, ev) in enumerate(eig_vec) if i in indices])
    print("eigen vector shape {}".format(eig_vec.shape))
    print(np.array2string(eig_vec))


dxy  = asl.df[features_dgnorm]
print("df shape {}".format(dxy.values.shape))
# wd = get_sliding_window(dxy,5,return2D=1)
# print("window shape {}".format(wd.shape))

# np.apply_along_axis(trajectory, axis = 1, arr=wd)


cor_mat = np.corrcoef(dxy.values.T)

eig_vals, eig_vecs = np.linalg.eig(cor_mat)

print('Eigenvectors shape {}'.format(eig_vecs.shape))
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
print('Everything ok!')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)


# with plt.style.context('seaborn-whitegrid'):
#     plt.figure(figsize=(6, 4))
#
#     plt.bar(range(4), var_exp, alpha=0.5, align='center',
#             label='individual explained variance')
#     plt.step(range(4), cum_var_exp, where='mid',
#              label='cumulative explained variance')
#     plt.ylabel('Explained variance ratio')
#     plt.xlabel('Principal components')
#     plt.legend(loc='best')
#     plt.tight_layout()
# plt.show()
# for x in np.nditer(wd):
#     print(x)
# y = dxy.rolling(window=5,center=True).apply(func=ev)

# asl.df[features_polar].rolling(window=5,center=True).apply(func=ev)

u,s,v  = np.linalg.svd(dxy.values.T)
print("svd")
print("{}".format(u))
# def transform_to_new_feature_space():

# eig_pairs = [(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
# eig_pairs.sort(key=lambda x:x[0],reverse=True)

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),eig_pairs[1][1].reshape(4,1)))
print("Matrix W:\n",matrix_w)

Y = dxy.values.dot(matrix_w)

print("Y sahpe:\n",Y.shape)
