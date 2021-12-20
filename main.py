
import datetime
from point import *
import cv2
import math
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import *
#数学后测4
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
#数据处理 ， 默认丢弃含有缺失值的行
# filename_list = ['Project63-57 Recording23','Project63-57 Recording24','Project63-57 Recording28',
#           'Project63-57 Recording30','Project63-57 Recording32',
#           'Project77-70 Recording18','Project77-70 Recording25','Project77-70 Recording26',
#           'Project77-70 Recording31','Project77-70 Recording46','Project77-70 Recording70']
filename_list = ['Project77-70 Recording46','Project77-70 Recording70']
X_sum = []#三维
X_all = []#二维
pre_means_all = []
pre_covs_all = []
lengths = []
def make_ellipses(mean, cov, ax, confidence=5.991, alpha=0.3, color="blue", eigv=False, arrow_color_list=None):
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
                     width=0.05,
                     length_includes_head=True,
                     head_width=0.2,
                     head_length=0.3,
                     color=arrow_color_list[i])
#'Project63-57 Recording63',
for filename in filename_list:
    in_dir='E://read-allquestion/hou_shu_01/'+filename+'.tsv'
    print("in_dir")
    print(in_dir)
    df = pd.read_csv(in_dir, sep='\t', header=0)

    component_num = 7 #隐藏状态数目
    mix_num = 5
    print("原始数据的大小：", df.shape)
    #print("原始数据的列名", df.columns)

    df.dropna(subset=['Gaze point X [MCS px]','Gaze point Y [MCS px]'], inplace = True)

    #df = pd.read_excel("C://Users/WANG/PycharmProjects/GMMHMM/SH.xlsx", header=0)
    #df.drop(['index','交易日期','开盘价','最高价','最低价' ,'市值', '换手率', 'pe', 'pb'], axis=1, inplace=True)

    gazeX = df['Gaze point X [MCS px]']
    gazeY = df['Gaze point Y [MCS px]']


    #获得输入数据
    X = np.column_stack([gazeX, gazeY])
    print("输入数据的大小：", X.shape)   #(1504, 2)
    #清洗数据，删掉超出屏幕范围的数据
    X_clear = []
    for i,element in enumerate(X):

        if element[0]>=0 and element[0]<=1920 and element[1]>=0 and element[1]<=1080:
            X_clear.append(element)
    X = np.array(X_clear)

    print("清洗后输入数据的大小：", X.shape)   #(1504, 2)
    # #计算均值和协方差
    # for x in X:
    #     if

    #X.T
    #[[ 899.  893.  899.  896.  930.  926.  927.  948.  938.  918.  879.  887.]
    # [1229. 1105. 1140. 1187. 1099. 1111. 1153. 1107. 1103. 1063.  929.  967.]]

    #用手动划分的AOI 初始化均值和协方差
    def in_which_AOI2(x,y) :#Statement合并
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
    X0=[]
    X1=[]
    X2=[]
    X3=[]
    X4=[]
    X5=[]
    X6=[]

    for i,element in enumerate(X):
        print(i, element)
        print("在哪个AOI中")
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

    X0 = np.array(X0)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    X4 = np.array(X4)
    X5 = np.array(X5)
    X6 = np.array(X6)

    print("X0")
    print(X0)
    print("X1")
    print(X1)
    print("X2")
    print(X2)

    # print("X.T")
    # print(X.T)
    cov_X0 = np.cov(X0.T)
    print("协方差X0")
    print(cov_X0)
    mean_X0 = np.mean(X0.T ,axis=1)
    print("均值X0")
    print(mean_X0)


    cov_X1 = np.cov(X1.T)
    print("协方差X1")
    print(cov_X1)
    mean_X1 = np.mean(X1.T ,axis=1)
    print("均值X1")
    print(mean_X1)


    cov_X2 = np.cov(X2.T)
    print("协方差X2")
    print(cov_X2)
    mean_X2 = np.mean(X2.T ,axis=1)
    print("均值X2")
    print(mean_X2)


    cov_X3 = np.cov(X3.T)
    print("协方差X3")
    print(cov_X3)
    mean_X3 = np.mean(X3.T ,axis=1)
    print("均值X3")
    print(mean_X3)


    cov_X4 = np.cov(X4.T)
    print("协方差X4")
    print(cov_X4)
    mean_X4 = np.mean(X4.T ,axis=1)
    print("均值X4")
    print(mean_X4)


    cov_X5 = np.cov(X5.T)
    print("协方差X5")
    print(cov_X5)
    mean_X5 = np.mean(X5.T ,axis=1)
    print("均值X5")
    print(mean_X5)


    cov_X6 = np.cov(X6.T)
    print("协方差X6")
    print(cov_X6)
    mean_X6 = np.mean(X6.T ,axis=1)
    print("均值X6")
    print(mean_X6)

    # #异常值的处理
    # min = X.mean(axis=0)[0] - 8*X.std(axis=0)[0]   #最小值
    # max = X.mean(axis=0)[0] + 8*X.std(axis=0)[0]  #最大值
    # X = pd.DataFrame(X)
    # #异常值设为均值
    # for i in range(len(X)):  #dataframe的遍历
    #     if (X.loc[i, 0]< min) | (X.loc[i, 0] > max):
    #             X.loc[i, 0] = X.mean(axis=0)[0]

    X = X.tolist()
    #模型的构建
    #数据集的划分
    #模型的搭建
    pre_means_ = [mean_X0,mean_X1, mean_X2, mean_X3, mean_X4, mean_X5, mean_X6]
    pre_covs_ = [cov_X0, cov_X1, cov_X2, cov_X3, cov_X4, cov_X5, cov_X6]
    pre_covs_ = np.array(pre_covs_)
    #X 接在X_all后面
    X_all += X
    X_sum.append(X)#三维数组，目的是后期根据观测序列运行Viterbi
    #多序列的均值和方差如何设置
    # pre_means_all+=pre_means_
    # pre_means_all.append(pre_means_)
    # pre_covs_all.append(pre_covs_)
    lengths.append(len(X))

print("X_all")
print(X_all)
X_all = np.array(X_all)

# pre_means_all = np.array(pre_means_all)
# pre_covs_all = np.array(pre_covs_all)
# lengths = np.array(lengths)

print("X_all")
print(X_all)

print("lengths")
print(lengths)

print("X_sum")
print(X_sum)
#model = GaussianHMM(n_components=component_num, covariance_type='full', n_iter=1000) #'stmcw', init_params='stmc'

model = GMMHMM(n_components=component_num, n_mix=mix_num, covariance_type='full', n_iter=1)  # , init_params='stmcw''stmcw'

# #在迭代过程中将点归回预先定义的AOI中
# for i in range(1000):#迭代次数
#     model = GMMHMM(n_components=component_num, n_mix=mix_num, covariance_type='full', n_iter=1, init_params='stmcw') #'stmcw'
#
#     model.fit(X_all, lengths)  # 拟合函数
#     #
#     print("均值矩阵")
#     print(model.means_)
#
#     for i in range(component_num):#一共7个状态
#         for j in range(mix_num):#一共5个混合成分
#             # x_y = (model.means_[i][0], model.means_[i][1])
#             if model.means_[i][j][0]
#                 model.means_[i][j][1]
#     print("end")

# model.means_ = pre_means_all #赋值初始矩阵
#
# # print("model.means_")
# # print(model.means_)
# # print("pre_covs_ ")
# # print(pre_covs_.shape)
# # print(pre_covs_)
# #model.covars_ = pre_covs_
# # #shape==2 diag???? 负值??
# #print(model.covars_.shape )
model.fit(X_all, lengths)#拟合函数
#
print("隐藏状态的个数", model.n_components)  #
print("均值矩阵")
print(model.means_)
print("协方差矩阵")
print(model.covars_)
print("状态转移矩阵--A")
print(model.transmat_)


#拟合观测序列
O_seq = np.array(X_sum[0])
print("O_seq")
print(O_seq)
state_sequence = model.predict(O_seq, lengths=None)
print("state_sequence")
print(state_sequence) #预测最可能的隐藏状态

np.savetxt('out/state_sequence.txt', state_sequence, fmt="%.3f", delimiter=',') #保存为3位小数的浮点数，用逗号分隔

print("predict_proba")
posterior_probability = model.predict_proba(O_seq, lengths=None)
print(posterior_probability)
np.savetxt('out/predict_proba.txt', posterior_probability, fmt="%.3f", delimiter=',') #保存为3位小数的浮点数，用逗号分隔

#计算信息熵
# shang = 0
# for row in posterior_probability:
#     for cell in row:
#         if cell!= 0:
#             shang += -cell*np.log(cell)
#         # posterior_probability[0]
# print("shang")
# print(shang)


# print("predict_proba")
# pp = model.predict_proba(X_all)
# print(pp)
# print("model.monitor_.converged")
# print(model.monitor_.converged)
# # cv2.putText(img, str(i), (int(model.means_[i][0]), int(model.means_[i][1])), cv2.FONT_ITALIC, 0.9, (210, 50, 220),
# #             2, cv2.LINE_AA)

#将拟合后的均值画在原始背景图上面
in_file = 'background.jpg'
# target_file = 'out/'+filename+'.png'
target_file = 'out/all.png'

orin_img = cv2.imread(in_file)
img = cv2.resize(orin_img, (1920, 1080))


# # 单高斯输出图片
# for i in range(component_num):
#         #x_y = (model.means_[i][0], model.means_[i][1])
#         cv2.circle(img, (int(model.means_[i][0]), int(model.means_[i][1])), 5, (0, 0, 255), -1)
#         cv2.putText(img, str(i), (int(model.means_[i][0]), int(model.means_[i][1])), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)


#高斯混合输出图片
for i in range(component_num):
    for j in range(mix_num):
        #x_y = (model.means_[i][0], model.means_[i][1])
        cv2.circle(img, (int(model.means_[i][j][0]), int(model.means_[i][j][1])), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i), (int(model.means_[i][j][0]), int(model.means_[i][j][1])), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)

cv2.imwrite(target_file, img)

# #需要归一化
# plt.rcParams["figure.figsize"] = (1.0, 1.0)
# fig, ax = plt.subplots()
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# confidence = 5.991
# color = "blue"
# alpha = 0.3
# eigv = False
# make_ellipses(model.means_, model.covars_, ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)
#
# plt.savefig("/out/gaussian_covariance_matrix.png")
# #plt.show()