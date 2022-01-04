import datetime
import pickle
from point import *
import cv2
import math
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import *
from sklearn.mixture import GaussianMixture

component_num = 6 #隐藏状态数目
mix_num = 4
iter_num = 10
for_num = 1
question_name = 'hou_shu_01'
#将拟合后的均值画在原始背景图上面，设置一些路径等参数
in_file = 'background.jpg'
# target_file = 'out/'+filename+'.png'
target_file1 = 'out/filemeansGMM-new8.png'
ini_file = 'out/inifilemeansGMM.png'
cov_ini_file = 'out/init_gaussian_covariance_matrix.png'
cov_file = 'out/gaussian_covariance_matrix.png'
modle_file = 'out/hmmmodel.pkl'

#数学后测1
diagram_LU = point(635, 57)
diagram_RB = point(1265, 557)

optionA_LU = point(833, 669)
optionA_RB = point(1085, 731)

optionB_LU = point(177, 741)
optionB_RB = point(1769, 821)

optionC_LU = point(177, 823)
optionC_RB = point(1769, 897)

optionD_LU = point(109, 903)
optionD_RB = point(1807, 1071)

option_LU = point(107, 681)
option_RB = point(1807, 1071)

stament_LU = point(105, 559)
stament_RB = point(1819, 661)

time_LU = point(981, 619)
time_RB = point(1111, 665)

width = 1920
height = 1080

# 用手动划分的AOI 初始化均值和协方差
def in_which_AOI2(x, y):  # Statement合并
    x = x * width
    y = y * height
    if x >= diagram_LU.x and x <= diagram_RB.x and y >= diagram_LU.y and y <= diagram_RB.y:
        return 0
    elif x >= optionA_LU.x and x <= optionA_RB.x and y >= optionA_LU.y and y <= optionA_RB.y:
        return 1
    elif x >= optionB_LU.x and x <= optionB_RB.x and y >= optionB_LU.y and y <= optionB_RB.y:
        return 2
    elif x >= optionC_LU.x and x <= optionC_RB.x and y >= optionC_LU.y and y <= optionC_RB.y:
        return 3
    elif x >= optionD_LU.x and x <= optionD_RB.x and y >= optionD_LU.y and y <= optionD_RB.y:
        return 4
    elif x >= time_LU.x and x <= time_RB.x and y >= time_LU.y and y <= time_RB.y:
        return 6
    elif x >= stament_LU.x and x <= stament_RB.x and y >= stament_LU.y and y <= stament_RB.y:
        return 5
    else:
        return 7

orin_img = cv2.imread(in_file)
img = cv2.resize(orin_img, (width, height))
imgini = cv2.resize(orin_img, (width, height))
#
#数据处理 ， 默认丢弃含有缺失值的行
filename_list = ['Project63-57 Recording23','Project63-57 Recording24','Project63-57 Recording28',
          'Project63-57 Recording30','Project63-57 Recording32','Project63-57 Recording63',
          'Project77-70 Recording18','Project77-70 Recording25','Project77-70 Recording26',
          'Project77-70 Recording31','Project77-70 Recording46','Project77-70 Recording70']
# filename_list = ['Project77-70 Recording46', 'Project77-70 Recording70']
X_sum = []#三维
X_all = []#二维
timestamp_sum = []#二维[[2333,2345,3214,...],[5333,5345,5214,...],...]
pre_means_all = []
pre_covs_all = []
lengths = []
#["red","blue","green","yellow","pink","black"]
def make_ellipses(str, mean, cov, ax, confidence=5.991, alpha=0.3, color="blue", eigv=True, arrow_color_list=None):
    """
    多元正态分布
    mean: 均值
    cov: 协方差矩阵
    ax: 画布的Axes对象
    confidence: 置信椭圆置信率 # 置信区间， 95%： 5.991  99%： 9.21  90%： 4.605
    alpha: 椭圆透明度
    eigv: 是否画特征向量
    arrow_color_list: 箭头颜色列表
    """
    lambda_, v = np.linalg.eig(cov)    # 计算特征值lambda_和特征向量v
    # print "lambda: ", lambda_
    # print "v: ", v
    # print "v[0, 0]: ", v[0, 0]

    sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值

    s = confidence
    width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
    height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
    angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
    ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆

    ax.add_artist(ell)
    ell.set_alpha(alpha)
    # 是否画出特征向量
    if eigv:
        # print "type(v): ", type(v)
        if arrow_color_list is None:
            arrow_color_list = [color for i in range(v.shape[0])]
        for i in range(v.shape[0]):
            v_i = v[:, i]
            scale_variable = np.sqrt(s) * sqrt_lambda[i]
            # 绘制箭头
            """
            ax.arrow(x, y, dx, dy,    # (x, y)为箭头起始坐标，(dx, dy)为偏移量
                     width,    # 箭头尾部线段宽度
                     length_includes_head,    # 长度是否包含箭头
                     head_width,    # 箭头宽度
                     head_length,    # 箭头长度
                     color,    # 箭头颜色
                     )
            """
            ax.arrow(mean[0], mean[1], scale_variable*v_i[0], scale_variable * v_i[1],
                     width=0.005,
                     length_includes_head=True,
                     head_width=0.02,
                     head_length=0.03,
                     color=arrow_color_list[i])
#'Project63-57 Recording63',

X0 = []
X1 = []
X2 = []
X3 = []
X4 = []
X5 = []
X6 = []
for filename in filename_list:
    in_dir = 'E://read-allquestion/'+question_name+'/'+filename+'.tsv'
    print("in_dir")
    print(in_dir)
    df = pd.read_csv(in_dir, sep='\t', header=0)

    print("原始数据的大小：", df.shape)
    #print("原始数据的列名", df.columns)

    df.dropna(subset=['Recording timestamp [ms]','Gaze point X [MCS px]','Gaze point Y [MCS px]'], inplace = True)

    timestamp = df['Recording timestamp [ms]']
    gazeX = df['Gaze point X [MCS px]']
    gazeY = df['Gaze point Y [MCS px]']

    timestamp = np.array(timestamp)
    #获得输入数据,数据归一化
    X = np.column_stack([gazeX/width, gazeY/height])
    print("输入数据的大小：", X.shape)   #(1504, 2)
    print("timestamp数据的大小：", timestamp.shape)   #(1504, 2)
    #清洗数据，删掉超出屏幕范围的数据
    X_clear = []
    timestamp_clear = []
    for i, element in enumerate(X):
        if element[0]>=0 and element[0]<=width and element[1]>=0 and element[1]<=height:
            X_clear.append(element)
            timestamp_clear.append(timestamp[i])

    X = np.array(X_clear)
    timestamp = np.array(timestamp_clear)
    print("清洗后输入数据的大小：", X.shape)   #(1504, 2)
    print("清洗后timestamp数据的大小：", timestamp.shape)   #(1504, 2)

    for i, element in enumerate(X):
        # print(i, element)
        # print("在哪个AOI中")
        if in_which_AOI2(element[0],element[1])==0:
            X0.append(element)
        elif in_which_AOI2(element[0],element[1])==1:
            X1.append(element)
        elif in_which_AOI2(element[0],element[1])==2:
            X2.append(element)
        elif in_which_AOI2(element[0],element[1])==3:
            X3.append(element)
        elif in_which_AOI2(element[0],element[1])==4:
            X4.append(element)
        elif in_which_AOI2(element[0],element[1])==5:
            X5.append(element)
        elif in_which_AOI2(element[0],element[1])==6:
            X6.append(element)

    X = X.tolist()
    timestamp = timestamp.tolist()
    X_all += X#二维数组，目的是后期根据观测序列运行Viterbi
    X_sum.append(X)#三维数组
    timestamp_sum.append(timestamp)#二维数组
    #多序列的均值和方差如何设置
    # pre_means_all += pre_means_
    # pre_means_all.append(pre_means_)
    lengths.append(len(X))


X0 = np.array(X0)
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)
X4 = np.array(X4)
X5 = np.array(X5)
X6 = np.array(X6)

gm0 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X0)#,covariance_type='full'
print("gm0.means_")
print(gm0.means_)
mean_X0 = gm0.means_
cov_X0 = gm0.covariances_
weight_X0 = gm0.weights_
print("mean_X0")
print(mean_X0)
print("cov_X0")
print(cov_X0)

gm1 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X1)
print("gm1.means_")
print(gm1.means_)
mean_X1 = gm1.means_
cov_X1 = gm1.covariances_
weight_X1 = gm1.weights_
print("mean_X1")
print(mean_X1)
print("cov_X1")
print(cov_X1)
print("mean_X1")
print(mean_X1)

gm2 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X2)
print("gm1.means_")
print(gm2.means_)
mean_X2 = gm2.means_
cov_X2 = gm2.covariances_
weight_X2 = gm2.weights_
print("mean_X2")
print(mean_X2)
print("cov_X2")
print(cov_X2)
print("mean_X2")
print(mean_X2)

gm3 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X3)
print("gm1.means_")
print(gm3.means_)
mean_X3 = gm3.means_
cov_X3 = gm3.covariances_
weight_X3 = gm3.weights_
print("mean_X3")
print(mean_X3)
print("cov_X3")
print(cov_X3)
print("mean_X3")
print(mean_X3)

gm4 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X4)
print("gm1.means_")
print(gm4.means_)
mean_X4 = gm4.means_
cov_X4 = gm4.covariances_
weight_X4 = gm4.weights_
print("mean_X4")
print(mean_X4)
print("cov_X4")
print(cov_X4)
print("mean_X4")
print(mean_X4)

gm5 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X5)#, covariance_type='full'
print("gm1.means_")
print(gm5.means_)
mean_X5 = gm5.means_
cov_X5 = gm5.covariances_
weight_X5 = gm5.weights_
print("mean_X5")
print(mean_X5)
print("cov_X5")
print(cov_X5)
print("mean_X5")
print(mean_X5)

gm6 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X6)
print("gm1.means_")
print(gm6.means_)
mean_X6 = gm6.means_
cov_X6 = gm6.covariances_
weight_X6 = gm6.weights_
print("mean_X6")
print(mean_X6)
print("cov_X6")
print(cov_X6)
print("mean_X6")
print(mean_X6)

pre_means_ = [mean_X0, mean_X1, mean_X2, mean_X3, mean_X4, mean_X5, mean_X6]  #
pre_covs_ = [cov_X0, cov_X1, cov_X2, cov_X3, cov_X4, cov_X5, cov_X6]  #
pre_weight_ = [weight_X0, weight_X1, weight_X2, weight_X3, weight_X4, weight_X5, weight_X6]
pre_means_ = np.array(pre_means_)
pre_covs_ = np.array(pre_covs_)
pre_weight_ = np.array(pre_weight_)
print("pre_means_")
print(pre_means_)
print("pre_covs_")
print(pre_covs_)

#看一下初始值 高斯混合输出图片
for i in range(component_num):
    for j in range(mix_num):
        #x_y = (model.means_[i][0], model.means_[i][1])
        cv2.circle(imgini, (int(pre_means_[i][j][0]*width), int(pre_means_[i][j][1]*height)), 5, (0, 0, 255), -1)
        cv2.putText(imgini, str(i)+str(j), (int(pre_means_[i][j][0]*width), int(pre_means_[i][j][1]*height)), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)
cv2.imwrite(ini_file, imgini)


#ini 输出置信度是0.95的高斯混合模型，需要归一化
plt.rcParams["figure.figsize"] = (10.0, 10.0)
fig, ax = plt.subplots()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(color='r', ls='dashed', lw=0.5, alpha=0.5) # 设置网格
confidence = 5.991
color = "red"
alpha = 0.3
eigv = True
for i in range(component_num):
    for j in range(mix_num):
        make_ellipses(str(i)+str(j), pre_means_[i][j], pre_covs_[i][j], ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)

plt.savefig(cov_ini_file)


X_all = np.array(X_all)

gm_list = [gm0, gm1, gm2, gm3, gm4, gm5, gm6]
# pre_means_all = np.array(pre_means_all)
# pre_covs_all = np.array(pre_covs_all)
# lengths = np.array(lengths)

# print("X_all")
# print(X_all)

p0 = np.exp(gm_list[0].score_samples(X_sum[0]))#p(x)
p1 = np.exp(gm_list[1].score_samples(X_sum[0]))#p(x)
p2 = np.exp(gm_list[2].score_samples(X_sum[0]))#p(x)
p3 = np.exp(gm_list[3].score_samples(X_sum[0]))#p(x)
p4 = np.exp(gm_list[4].score_samples(X_sum[0]))#p(x)
p5 = np.exp(gm_list[5].score_samples(X_sum[0]))#p(x)
print("p")

# p0 = gm_list[0].score_samples(X_sum[0])#p(x)
# p1 = gm_list[1].score_samples(X_sum[0])#p(x)
# p2 = gm_list[2].score_samples(X_sum[0])#p(x)
# p3 = gm_list[3].score_samples(X_sum[0])#p(x)
# p4 = gm_list[4].score_samples(X_sum[0])#p(x)
# p5 = gm_list[5].score_samples(X_sum[0])#p(x)
p_all = []
AOI_distribute = [] #输入朱老师算法的矩阵
for j in range(len(X_sum)):#第j个人的数据 X_sum[j]
    a_AOI_distribute = []
    p = []
    # p_sum = 0
    for i in range(component_num):#第i个高斯
        pi = np.exp(gm_list[i].score_samples(X_sum[j]))#p(x)
        p.append(pi)
        #p_sum += pi
    #     print("end")
    # print(p)
    p_all.append(p)
    for i in range(len(X_sum[j])):
        AOI_distribute_frame = [] #每一帧的AOI概率分布
        p_sum = 0 #p(x|l)
        for k in range(component_num):
            p_sum += p[k][i]

        for m in range(component_num):
            AOI_distribute_frame.append(p[m][i] / p_sum)
        a_AOI_distribute.append(AOI_distribute_frame)
    print(j)

    AOI_distribute.append(a_AOI_distribute)
    # print(p_sum)

print("AOI_distribute")
print(AOI_distribute)

#将AOI_distribute输入到文档里面作为朱老师算法输入
for i in range(len(AOI_distribute)):
    np.savetxt('out/'+filename_list[i]+'.txt', AOI_distribute[i], fmt="%.3f", delimiter=',')