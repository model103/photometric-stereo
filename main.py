import numpy as np
import cv2
import glob
from sklearn.preprocessing import normalize
import imageio
import scipy.sparse as sp
import progress_bar

for i in range(4):
    ima = imageio.imread('../光度立体/im' + str(i + 1) + '.png')
# np.save('../光度立体/mpy/im' + str(i+1), ima)

# 光源位置数据路径以及物体各帧图像的路径
# lights_path = r'.\RobustPhotometricStereo\data\bunny\lights.npy'
# bunny_path = r'.\光度立体\mpy\*.npy'
# bunny_path = '../光度立体/npy/*.npy'

# 读取光源方向
# 读取到的是 L.T
# Lt = np.load(lights_path)
Lt = np.array([[0, 0, -1], [0, 0.2, -1], [0, -0.2, -1], [0.2, 0, -1]])
# print(Lt.shape)

# 读取图像

M = []
"""
for fname in sorted(glob.glob(bunny_path)):
    print("test")
    im = np.load(fname).astype(np.float32)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    if M == []:
        height, width = im.shape
        M = im.reshape((-1, 1))
    else:
        M = np.append(M, im.reshape((-1, 1)), axis=1)
"""
for i in range(4):

    path = 'im' + str(i + 1) + '.png'
    # print(path)
    # im = np.load('..\光度立体\mpy\im'+ str(i+1) + '.npy').astype(np.float32)
    # im = cv2.cvtColor(im) #, cv2.COLOR_RGB2GRAY
    im = cv2.imread(path, 0)
    # print(im)
    if len(M) == 0:
        height, width = im.shape
        M = im.reshape((-1, 1))
    else:
        M = np.append(M, im.reshape((-1, 1)), axis=1)
M = np.asarray(M)

# 光度立体计算 使用最小二乘法
# M = NL <-> M.T = L.T N.T
N = np.linalg.lstsq(Lt, M.T, rcond=None)[0].T
N = normalize(N, axis=1)

N = np.reshape(N, (height, width, 3))
N[:, :, 0], N[:, :, 2] = N[:, :, 2], N[:, :, 0].copy()
N = (N + 1.0) / 2.0  # 得平面法向量N

mask = np.zeros((100, 100))
# print(mask.shape)
A = sp.lil_matrix((mask.size * 2, mask.size))  # 10*5 拼接
# print('A',A.shape)
b = np.zeros(A.shape[0], dtype=np.float32)  # 200*1

# 2. set normal
nx = N[:, :, 0].ravel()  # 取量向量第一个坐标 100*100
ny = N[:, :, 1].ravel()
nz = N[:, :, 2].ravel()


b[0:nx.shape[0]] = -ny / nz  # b[0:100]指取b从第0行到99行
b[nx.shape[0]:b.shape[0]] = -nx / nz  # 取b的第100行到199行

dif = mask.size
# print(dif)
w = mask.shape[1]
# print(w)
h = mask.shape[0]

for i in range(mask.shape[0]):
    # progress_bar(i, mask.shape[0] -1)
    for j in range(mask.shape[1]):
        # current pixel om matrix
        pixel = (i * w) + j

        # for v1(right)
        if j != w - 1:
            A[pixel, pixel] = -1
            A[pixel, pixel + 1] = 1

        # for v2(up)
        if i != h - 1:
            A[pixel + dif, pixel] = -1
            A[pixel + dif, pixel + w] = 1

# print(A)

'''
AtA = A.transpose().dot(A)
Atb = A.transpose().dot(b)
x, info = sp.linalg.cg(AtA, Atb)

#print(x)

x_max = np.max(x)
x_min = np.min(x)
for i in range(10000):
    x[i] = (x[i] - x_min) / (x_max - x_min) * 255

x = np.reshape(x, (100, 100))

#print(x)
'''

AtA = A.T @ A

Atb = A.T @ b

z = sp.linalg.spsolve(AtA, Atb)
std_z = np.std(z, ddof=1)
mean_z = np.mean(z)
z_zscore = (z - mean_z) / std_z
# 因奇异值造成的异常
outlier_ind = np.abs(z_zscore) > 10
z_min = np.min(z[~outlier_ind])
z_max = np.max(z[~outlier_ind])
# 将z填充回正常的2D形状
Z = mask.astype('float')
for idx in range(10000):
    h = idx // 100
    w = idx % 100
    Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255

cv2.imshow('normal map', Z)
cv2.waitKey()
cv2.destroyAllWindows()
