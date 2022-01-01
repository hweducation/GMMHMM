# import datetime
# import pickle
#
# from point import *
# import cv2
# import math
# import numpy as np
# import pandas as pd
#
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# from matplotlib import cm, pyplot as plt
# from hmmlearn.hmm import *
# from sklearn.mixture import GaussianMixture
#
# component_num = 6 #隐藏状态数目
# mix_num = 4
# iter_num = 10
# for_num = 1
# question_name = 'hou_shu_01'
# #将拟合后的均值画在原始背景图上面，设置一些路径等参数
# in_file = 'background.jpg'
# # target_file = 'out/'+filename+'.png'
# target_file1 = 'out/filemeansGMM-new8.png'
# ini_file = 'out/inifilemeansGMM.png'
# cov_ini_file = 'out/init_gaussian_covariance_matrix.png'
# cov_file = 'out/gaussian_covariance_matrix.png'
# modle_file = 'out/hmmmodel.pkl'
#
#
# #数学后测1
# diagram_LU = point(635, 57)
# diagram_RB = point(1265, 557)
#
# optionA_LU = point(833, 669)
# optionA_RB = point(1085, 731)
#
# optionB_LU = point(177, 741)
# optionB_RB = point(1769, 821)
#
# optionC_LU = point(177, 823)
# optionC_RB = point(1769, 897)
#
# optionD_LU = point(109, 903)
# optionD_RB = point(1807, 1071)
#
# option_LU = point(107, 681)
# option_RB = point(1807, 1071)
#
# stament_LU = point(105, 559)
# stament_RB = point(1819, 661)
#
# time_LU = point(981, 619)
# time_RB = point(1111, 665)
#
# width = 1920
# height = 1080
#
#
# # 用手动划分的AOI 初始化均值和协方差
# def in_which_AOI2(x, y):  # Statement合并
#     x = x * width
#     y = y * height
#     if x >= diagram_LU.x and x <= diagram_RB.x and y >= diagram_LU.y and y <= diagram_RB.y:
#         return 0
#     elif x >= optionA_LU.x and x <= optionA_RB.x and y >= optionA_LU.y and y <= optionA_RB.y:
#         return 1
#     elif x >= optionB_LU.x and x <= optionB_RB.x and y >= optionB_LU.y and y <= optionB_RB.y:
#         return 2
#     elif x >= optionC_LU.x and x <= optionC_RB.x and y >= optionC_LU.y and y <= optionC_RB.y:
#         return 3
#     elif x >= optionD_LU.x and x <= optionD_RB.x and y >= optionD_LU.y and y <= optionD_RB.y:
#         return 4
#     elif x >= time_LU.x and x <= time_RB.x and y >= time_LU.y and y <= time_RB.y:
#         return 6
#     elif x >= stament_LU.x and x <= stament_RB.x and y >= stament_LU.y and y <= stament_RB.y:
#         return 5
#     else:
#         return 7
#
# def to_edge(x, y, index):  # 将跑出AOI的点归到边缘
#     x = x * width
#     y = y * height
#     new_x = x
#     new_y = y
#     if index == 0:#将x
#         if x < diagram_LU.x:
#             new_x = diagram_LU.x
#         elif x > diagram_RB.x:
#             new_x = diagram_RB.x
#
#         if y > diagram_LU.y:
#             new_y = diagram_LU.y
#         elif y < diagram_RB.y:
#             new_y = diagram_RB.y
#     elif index == 1:
#         if x < optionA_LU.x:
#             new_x = optionA_LU.x
#         elif x > optionA_RB.x:
#             new_x = optionA_RB.x
#
#         if y > optionA_LU.y:
#             new_y = optionA_LU.y
#         elif y < optionA_RB.y:
#             new_y = optionA_RB.y
#
#     elif index == 2:
#         if x < optionB_LU.x:
#             new_x = optionB_LU.x
#         elif x > optionB_RB.x:
#             new_x = optionB_RB.x
#
#         if y > optionB_LU.y:
#             new_y = optionB_LU.y
#         elif y < optionB_RB.y:
#             new_y = optionB_RB.y
#
#     elif index == 3:
#         if x < optionC_LU.x:
#             new_x = optionC_LU.x
#         elif x > optionC_RB.x:
#             new_x = optionC_RB.x
#
#         if y > optionC_LU.y:
#             new_y = optionC_LU.y
#         elif y < optionC_RB.y:
#             new_y = optionC_RB.y
#
#     elif index == 4:
#         if x < optionD_LU.x:
#             new_x = optionD_LU.x
#         elif x > optionD_RB.x:
#             new_x = optionD_RB.x
#
#         if y > optionD_LU.y:
#             new_y = optionD_LU.y
#         elif y < optionD_RB.y:
#             new_y = optionD_RB.y
#     elif index == 5:
#         if x < stament_LU.x:
#             new_x = stament_LU.x
#         elif x > stament_RB.x:
#             new_x = stament_RB.x
#
#         if y > stament_LU.y:
#             new_y = stament_LU.y
#         elif y < stament_RB.y:
#             new_y = stament_RB.y
#     elif index == 6:
#         if x < time_LU.x:
#             new_x = time_LU.x
#         elif x > time_RB.x:
#             new_x = time_RB.x
#
#         if y > time_LU.y:
#             new_y = time_LU.y
#         elif y < time_RB.y:
#             new_y = time_RB.y
#     print("new_x")
#     print(new_x)
#     print(new_y)
#     return [new_x/width, new_y/height]
#
#
# orin_img = cv2.imread(in_file)
# img = cv2.resize(orin_img, (width, height))
# imgini = cv2.resize(orin_img, (width, height))
# #
# #数据处理 ， 默认丢弃含有缺失值的行
# filename_list = ['Project63-57 Recording23','Project63-57 Recording24','Project63-57 Recording28',
#           'Project63-57 Recording30','Project63-57 Recording32','Project63-57 Recording63',
#           'Project77-70 Recording18','Project77-70 Recording25','Project77-70 Recording26',
#           'Project77-70 Recording31','Project77-70 Recording46','Project77-70 Recording70']
# # filename_list = ['Project77-70 Recording46', 'Project77-70 Recording70']
# X_sum = []#三维
# X_all = []#二维
# timestamp_sum = []#二维[[2333,2345,3214,...],[5333,5345,5214,...],...]
# pre_means_all = []
# pre_covs_all = []
# lengths = []
# #["red","blue","green","yellow","pink","black"]
# def make_ellipses(str, mean, cov, ax, confidence=5.991, alpha=0.3, color="blue", eigv=True, arrow_color_list=None):
#     """
#     多元正态分布
#     mean: 均值
#     cov: 协方差矩阵
#     ax: 画布的Axes对象
#     confidence: 置信椭圆置信率 # 置信区间， 95%： 5.991  99%： 9.21  90%： 4.605
#     alpha: 椭圆透明度
#     eigv: 是否画特征向量
#     arrow_color_list: 箭头颜色列表
#     """
#     lambda_, v = np.linalg.eig(cov)    # 计算特征值lambda_和特征向量v
#     # print "lambda: ", lambda_
#     # print "v: ", v
#     # print "v[0, 0]: ", v[0, 0]
#
#     sqrt_lambda = np.sqrt(np.abs(lambda_))    # 存在负的特征值， 无法开方，取绝对值
#
#     s = confidence
#     width = 2 * np.sqrt(s) * sqrt_lambda[0]    # 计算椭圆的两倍长轴
#     height = 2 * np.sqrt(s) * sqrt_lambda[1]   # 计算椭圆的两倍短轴
#     angle = np.rad2deg(np.arccos(v[0, 0]))    # 计算椭圆的旋转角度
#     ell = mpl.patches.Ellipse(xy=mean, width=width, height=height, angle=angle, color=color)    # 绘制椭圆
#
#     ax.add_artist(ell)
#     ell.set_alpha(alpha)
#     # 是否画出特征向量
#     if eigv:
#         # print "type(v): ", type(v)
#         if arrow_color_list is None:
#             arrow_color_list = [color for i in range(v.shape[0])]
#         for i in range(v.shape[0]):
#             v_i = v[:, i]
#             scale_variable = np.sqrt(s) * sqrt_lambda[i]
#             # 绘制箭头
#             """
#             ax.arrow(x, y, dx, dy,    # (x, y)为箭头起始坐标，(dx, dy)为偏移量
#                      width,    # 箭头尾部线段宽度
#                      length_includes_head,    # 长度是否包含箭头
#                      head_width,    # 箭头宽度
#                      head_length,    # 箭头长度
#                      color,    # 箭头颜色
#                      )
#             """
#             ax.arrow(mean[0], mean[1], scale_variable*v_i[0], scale_variable * v_i[1],
#                      width=0.005,
#                      length_includes_head=True,
#                      head_width=0.02,
#                      head_length=0.03,
#                      color=arrow_color_list[i])
# #'Project63-57 Recording63',
#
# X0 = []
# X1 = []
# X2 = []
# X3 = []
# X4 = []
# X5 = []
# X6 = []
# for filename in filename_list:
#     in_dir = 'E://read-allquestion/'+question_name+'/'+filename+'.tsv'
#     print("in_dir")
#     print(in_dir)
#     df = pd.read_csv(in_dir, sep='\t', header=0)
#
#     print("原始数据的大小：", df.shape)
#     #print("原始数据的列名", df.columns)
#
#     df.dropna(subset=['Recording timestamp [ms]','Gaze point X [MCS px]','Gaze point Y [MCS px]'], inplace = True)
#
#     timestamp = df['Recording timestamp [ms]']
#     gazeX = df['Gaze point X [MCS px]']
#     gazeY = df['Gaze point Y [MCS px]']
#
#     timestamp = np.array(timestamp)
#     #获得输入数据,数据归一化
#     X = np.column_stack([gazeX/width, gazeY/height])
#     print("输入数据的大小：", X.shape)   #(1504, 2)
#     print("timestamp数据的大小：", timestamp.shape)   #(1504, 2)
#     #清洗数据，删掉超出屏幕范围的数据
#     X_clear = []
#     timestamp_clear = []
#     for i, element in enumerate(X):
#         if element[0]>=0 and element[0]<=width and element[1]>=0 and element[1]<=height:
#             X_clear.append(element)
#             timestamp_clear.append(timestamp[i])
#
#     X = np.array(X_clear)
#     timestamp = np.array(timestamp_clear)
#     print("清洗后输入数据的大小：", X.shape)   #(1504, 2)
#     print("清洗后timestamp数据的大小：", timestamp.shape)   #(1504, 2)
#     # #计算均值和协方差
#     # for x in X:
#     #     if
#
#     #X.T
#     #[[ 899.  893.  899.  896.  930.  926.  927.  948.  938.  918.  879.  887.]
#     # [1229. 1105. 1140. 1187. 1099. 1111. 1153. 1107. 1103. 1063.  929.  967.]]
#
#
#     for i, element in enumerate(X):
#         # print(i, element)
#         # print("在哪个AOI中")
#         if in_which_AOI2(element[0],element[1])==0:
#             X0.append(element)
#         elif in_which_AOI2(element[0],element[1])==1:
#             X1.append(element)
#         elif in_which_AOI2(element[0],element[1])==2:
#             X2.append(element)
#         elif in_which_AOI2(element[0],element[1])==3:
#             X3.append(element)
#         elif in_which_AOI2(element[0],element[1])==4:
#             X4.append(element)
#         elif in_which_AOI2(element[0],element[1])==5:
#             X5.append(element)
#         # elif in_which_AOI2(element[0],element[1])==6:
#         #     X6.append(element)
#     # cov_X6 = np.cov(X6.T)
#     # print("协方差X6")
#     # print(cov_X6)
#     # mean_X6 = np.mean(X6.T ,axis=1)
#     # mean_X6L=[mean_X6[0]-0.1, mean_X6[1]]
#     # mean_X6R=[mean_X6[0]+0.1, mean_X6[1]]
#     # mean_X6= [mean_X6, mean_X6L, mean_X6R]
#     # print("均值X6")
#     # print(mean_X6)
#     # timex1 = np.random.randint(time_LU.x, time_RB.x)/width
#     # timey1 = np.random.randint(time_LU.y, time_RB.y)/height
#     # timex2 = np.random.randint(time_LU.x, time_RB.x)/width
#     # timey2 = np.random.randint(time_LU.y, time_RB.y)/height
#     # timex3 = np.random.randint(time_LU.x, time_RB.x)/width
#     # timey3 = np.random.randint(time_LU.y, time_RB.y)/height
#     # time1 = [timex1,timey1]
#     # time2 = [timex2,timey2]
#     # time3 = [timex3,timey3]
#     # mean_X6 = [time1,time2,time3]
#
#     # gm6 = GaussianMixture(n_components=mix_num, random_state=0).fit(X6)
#     # print("gm1.means_")
#     # print(gm6.means_)
#     # mean_X6 = gm6.means_
#     # cov_X6 = gm6.covariances_
#     # print("mean_X6")
#     # print(mean_X6)
#     # print("cov_X6")
#     # print(cov_X6)
#     # print("mean_X6")
#     # print(mean_X6)
#
#     # #异常值的处理
#     # min = X.mean(axis=0)[0] - 8*X.std(axis=0)[0]   #最小值
#     # max = X.mean(axis=0)[0] + 8*X.std(axis=0)[0]  #最大值
#     # X = pd.DataFrame(X)
#     # #异常值设为均值
#     # for i in range(len(X)):  #dataframe的遍历
#     #     if (X.loc[i, 0]< min) | (X.loc[i, 0] > max):
#     #             X.loc[i, 0] = X.mean(axis=0)[0]
#     #X 接在X_all后面
#
#     X = X.tolist()
#     timestamp = timestamp.tolist()
#     X_all += X#二维数组，目的是后期根据观测序列运行Viterbi
#     X_sum.append(X)#三维数组
#     timestamp_sum.append(timestamp)#二维数组
#     #多序列的均值和方差如何设置
#     # pre_means_all += pre_means_
#     # pre_means_all.append(pre_means_)
#     lengths.append(len(X))
#
# # print("timestamp_sum")
# # print(timestamp_sum)
# X0 = np.array(X0)
# X1 = np.array(X1)
# X2 = np.array(X2)
# X3 = np.array(X3)
# X4 = np.array(X4)
# X5 = np.array(X5)
# # X6 = np.array(X6)
#
# # print("X0")
# # print(X0)
# # print("X1")
# # print(X1)
# # print("X2")
# # print(X2)
# # labels = gmm0.predict(X0)
#
# # print("X.T")
# # print(X.T)
# # cov_X0 = np.cov(X0.T)
# # print("协方差X0")
# # print(cov_X0)
# # mean_X0 = np.mean(X0.T ,axis=1)
# # mean_X0L=[mean_X0[0]-0.1, mean_X0[1]]
# # mean_X0R=[mean_X0[0]+0.1, mean_X0[1]]
# # mean_X0= [mean_X0,mean_X0L,mean_X0R]
# # # print("均值X0")
# # # print(mean_X0)
# # diagramx1 = np.random.randint(diagram_LU.x, diagram_RB.x)/width
# # diagramy1 = np.random.randint(diagram_LU.y, diagram_RB.y)/height
# # diagramx2 = np.random.randint(diagram_LU.x, diagram_RB.x)/width
# # diagramy2 = np.random.randint(diagram_LU.y, diagram_RB.y)/height
# # diagramx3 = np.random.randint(diagram_LU.x, diagram_RB.x)/width
# # diagramy3 = np.random.randint(diagram_LU.y, diagram_RB.y)/height
# # diagram1 = [diagramx1,diagramy1]
# # diagram2 = [diagramx2,diagramy2]
# # diagram3 = [diagramx3,diagramy3]
# # mean_X0 = [diagram1,diagram2,diagram3]
# gm0 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X0)#,covariance_type='full'
# print("gm0.means_")
# print(gm0.means_)
# mean_X0 = gm0.means_
# cov_X0 = gm0.covariances_
# weight_X0 = gm0.weights_
# print("mean_X0")
# print(mean_X0)
# print("cov_X0")
# print(cov_X0)
#
# # cov_X1 = np.cov(X1.T)
# # print("协方差X1")
# # print(cov_X1)
# # mean_X1 = np.mean(X1.T ,axis=1)
# # mean_X1L=[mean_X1[0]-0.1, mean_X1[1]]
# # mean_X1R=[mean_X1[0]+0.1, mean_X1[1]]
# # mean_X1= [mean_X1, mean_X1L, mean_X1R]
# # # print("均值X1")
# # # print(mean_X1)
# # optionAx1 = np.random.randint(optionA_LU.x, optionA_RB.x)/width
# # optionAy1 = np.random.randint(optionA_LU.y, optionA_RB.y)/height
# # optionAx2 = np.random.randint(optionA_LU.x, optionA_RB.x)/width
# # optionAy2 = np.random.randint(optionA_LU.y, optionA_RB.y)/height
# # optionAx3 = np.random.randint(optionA_LU.x, optionA_RB.x)/width
# # optionAy3 = np.random.randint(optionA_LU.y, optionA_RB.y)/height
# # optionA1 = [optionAx1,optionAy1]
# # optionA2 = [optionAx2,optionAy2]
# # optionA3 = [optionAx3,optionAy3]
# # mean_X1 = [optionA1,optionA2,optionA3]
#
# gm1 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X1)
# print("gm1.means_")
# print(gm1.means_)
# mean_X1 = gm1.means_
# cov_X1 = gm1.covariances_
# weight_X1 = gm1.weights_
# print("mean_X1")
# print(mean_X1)
# print("cov_X1")
# print(cov_X1)
# print("mean_X1")
# print(mean_X1)
#
# # cov_X2 = np.cov(X2.T)
# # print("协方差X2")
# # print(cov_X2)
# # mean_X2 = np.mean(X2.T ,axis=1)
# # mean_X2L=[mean_X2[0]-0.1, mean_X2[1]]
# # mean_X2R=[mean_X2[0]+0.1, mean_X2[1]]
# # mean_X2= [mean_X2, mean_X2L, mean_X2R]
# # # print("均值X2")
# # # print(mean_X2)
# # optionBx1 = np.random.randint(optionB_LU.x, optionB_RB.x)/width
# # optionBy1 = np.random.randint(optionB_LU.y, optionB_RB.y)/height
# # optionBx2 = np.random.randint(optionB_LU.x, optionB_RB.x)/width
# # optionBy2 = np.random.randint(optionB_LU.y, optionB_RB.y)/height
# # optionBx3 = np.random.randint(optionB_LU.x, optionB_RB.x)/width
# # optionBy3 = np.random.randint(optionB_LU.y, optionB_RB.y)/height
# # optionB1 = [optionBx1,optionBy1]
# # optionB2 = [optionBx2,optionBy2]
# # optionB3 = [optionBx3,optionBy3]
# # mean_X2 = [optionB1,optionB2,optionB3]
#
# gm2 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X2)
# print("gm1.means_")
# print(gm2.means_)
# mean_X2 = gm2.means_
# cov_X2 = gm2.covariances_
# weight_X2 = gm2.weights_
# print("mean_X2")
# print(mean_X2)
# print("cov_X2")
# print(cov_X2)
# print("mean_X2")
# print(mean_X2)
#
# # cov_X3 = np.cov(X3.T)
# # print("协方差X3")
# # print(cov_X3)
# # mean_X3 = np.mean(X3.T ,axis=1)
# # mean_X3L=[mean_X3[0]-0.1, mean_X3[1]]
# # mean_X3R=[mean_X3[0]+0.1, mean_X3[1]]
# # mean_X3= [mean_X3, mean_X3L, mean_X3R]
# # # print("均值X3")
# # # print(mean_X3)
# # optionCx1 = np.random.randint(optionC_LU.x, optionC_RB.x)/width
# # optionCy1 = np.random.randint(optionC_LU.y, optionC_RB.y)/height
# # optionCx2 = np.random.randint(optionC_LU.x, optionC_RB.x)/width
# # optionCy2 = np.random.randint(optionC_LU.y, optionC_RB.y)/height
# # optionCx3 = np.random.randint(optionC_LU.x, optionC_RB.x)/width
# # optionCy3 = np.random.randint(optionC_LU.y, optionC_RB.y)/height
# # optionC1 = [optionCx1,optionCy1]
# # optionC2 = [optionCx2,optionCy2]
# # optionC3 = [optionCx3,optionCy3]
# # mean_X3 = [optionC1,optionC2,optionC3]
#
# gm3 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X3)
# print("gm1.means_")
# print(gm3.means_)
# mean_X3 = gm3.means_
# cov_X3 = gm3.covariances_
# weight_X3 = gm3.weights_
# print("mean_X3")
# print(mean_X3)
# print("cov_X3")
# print(cov_X3)
# print("mean_X3")
# print(mean_X3)
#
# # cov_X4 = np.cov(X4.T)
# # print("协方差X4")
# # print(cov_X4)
# # mean_X4 = np.mean(X4.T ,axis=1)
# # mean_X4L=[mean_X4[0]-0.1, mean_X4[1]]
# # mean_X4R=[mean_X4[0]+0.1, mean_X4[1]]
# # mean_X4= [mean_X4, mean_X4L, mean_X4R]
# # # print("均值X4")
# # # print(mean_X4)
# # optionDx1 = np.random.randint(optionD_LU.x, optionD_RB.x)/width
# # optionDy1 = np.random.randint(optionD_LU.y, optionD_RB.y)/height
# # optionDx2 = np.random.randint(optionD_LU.x, optionD_RB.x)/width
# # optionDy2 = np.random.randint(optionD_LU.y, optionD_RB.y)/height
# # optionDx3 = np.random.randint(optionD_LU.x, optionD_RB.x)/width
# # optionDy3 = np.random.randint(optionD_LU.y, optionD_RB.y)/height
# # optionD1 = [optionDx1,optionDy1]
# # optionD2 = [optionDx2,optionDy2]
# # optionD3 = [optionDx3,optionDy3]
# # mean_X4 = [optionD1,optionD2,optionD3]
#
# gm4 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X4)
# print("gm1.means_")
# print(gm4.means_)
# mean_X4 = gm4.means_
# cov_X4 = gm4.covariances_
# weight_X4 = gm4.weights_
# print("mean_X4")
# print(mean_X4)
# print("cov_X4")
# print(cov_X4)
# print("mean_X4")
# print(mean_X4)
#
# # cov_X5 = np.cov(X5.T)
# # print("协方差X5")
# # print(cov_X5)
# # mean_X5 = np.mean(X5.T ,axis=1)
# # mean_X5L=[mean_X5[0]-0.1, mean_X5[1]]
# # mean_X5R=[mean_X5[0]+0.1, mean_X5[1]]
# # mean_X5= [mean_X5, mean_X5L, mean_X5R]
# # # print("均值X5")
# # # print(mean_X5)
# # stamentx1 = np.random.randint(stament_LU.x, stament_RB.x)/width
# # stamenty1 = np.random.randint(stament_LU.y, stament_RB.y)/height
# # stamentx2 = np.random.randint(stament_LU.x, stament_RB.x)/width
# # stamenty2 = np.random.randint(stament_LU.y, stament_RB.y)/height
# # stamentx3 = np.random.randint(stament_LU.x, stament_RB.x)/width
# # stamenty3 = np.random.randint(stament_LU.y, stament_RB.y)/height
# # stament1 = [stamentx1,stamenty1]
# # stament2 = [stamentx2,stamenty2]
# # stament3 = [stamentx3,stamenty3]
# # mean_X4 = [stament1,stament2,stament3]
#
# gm5 = GaussianMixture(n_components=mix_num, random_state=0, covariance_type='full').fit(X5)#, covariance_type='full'
# print("gm1.means_")
# print(gm5.means_)
# mean_X5 = gm5.means_
# cov_X5 = gm5.covariances_
# weight_X5 = gm5.weights_
# print("mean_X5")
# print(mean_X5)
# print("cov_X5")
# print(cov_X5)
# print("mean_X5")
# print(mean_X5)
#
# pre_means_ = [mean_X0, mean_X1, mean_X2, mean_X3, mean_X4, mean_X5]  # , mean_X6
# pre_covs_ = [cov_X0, cov_X1, cov_X2, cov_X3, cov_X4, cov_X5]  # , cov_X6
# pre_weight_ = [weight_X0, weight_X1, weight_X2, weight_X3, weight_X4, weight_X5]
# pre_means_ = np.array(pre_means_)
# pre_weight_ = np.array(pre_weight_)
# pre_covs_ = np.array(pre_covs_)
#
# print("pre_means_")
# print(pre_means_)
#
#
# print("pre_covs_")
# print(pre_covs_)
# #看一下初始值 高斯混合输出图片
# for i in range(component_num):
#     for j in range(mix_num):
#         #x_y = (model.means_[i][0], model.means_[i][1])
#         cv2.circle(imgini, (int(pre_means_[i][j][0]*width), int(pre_means_[i][j][1]*height)), 5, (0, 0, 255), -1)
#         cv2.putText(imgini, str(i)+str(j), (int(pre_means_[i][j][0]*width), int(pre_means_[i][j][1]*height)), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)
# cv2.imwrite(ini_file, imgini)
#
#
# #ini 输出置信度是0.95的高斯混合模型，需要归一化
# plt.rcParams["figure.figsize"] = (10.0, 10.0)
# fig, ax = plt.subplots()
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.grid(color='r', ls='dashed', lw=0.5, alpha=0.5) # 设置网格
# confidence = 5.991
# color = "red"
# alpha = 0.3
# eigv = True
# for i in range(component_num):
#     for j in range(mix_num):
#         make_ellipses(str(i)+str(j), pre_means_[i][j], pre_covs_[i][j], ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)
#
# plt.savefig(cov_ini_file)
#
# #plt.savefig("/out/gaussian_covariance_matrix.png")
# #plt.show()
#
# # print("X_all")
# # print(X_all)
# X_all = np.array(X_all)
#
# # pre_means_all = np.array(pre_means_all)
# # pre_covs_all = np.array(pre_covs_all)
# # lengths = np.array(lengths)
#
# # print("X_all")
# # print(X_all)
#
# print("lengths")
# print(lengths)
# #
# # print("X_sum")
# # print(X_sum)
# model = GMMHMM(n_components=component_num, n_mix=mix_num, covariance_type='full', n_iter = iter_num, verbose=True, init_params='st') #'stmcw'
#
# #给一个初值
# model.means_ =pre_means_  #赋值初始均值矩阵???
# model.covars_ = pre_covs_
# model.weights_ = pre_weight_
# #model.covars_ = pre_covs_
#
# # model = GMMHMM(n_components=component_num, n_mix=mix_num, covariance_type='full', n_iter=1)  # , init_params='stmcw''stmcw'
#
# # model = GMMHMM(n_components=component_num, n_mix=mix_num, covariance_type='full', n_iter=5,
# #                init_params='stmcw', verbose=True)  # 'stmcw'
# # model.means_ = pre_means_all  #赋值初始均值矩阵
#
# # print("model.means_")
# # print(model.means_)
# # print("pre_covs_ ")
# # print(pre_covs_.shape)
# # print(pre_covs_)
# #model.covars_ = pre_covs_
#
# model.fit(X_all, lengths)  # 拟合函数
#
# #在迭代过程中将点归回预先定义的AOI中,GMM
# for i in range(for_num): #迭代次数
#     model.fit(X_all, lengths)  # 拟合函数
#     print("均值矩阵")
#     print(model.means_)
#     print("协方差矩阵")
#     print(model.covars_)
#
#     new_means = []
#     new_means0 = []
#     new_means1 = []
#     new_means2 = []
#     new_means3 = []
#     new_means4 = []
#     new_means5 = []
#     #new_means6 = []
#     # if in_which_AOI2(model.means_[0][0],model.means_[0][1])!=0:
#     #     new_means.append(model.means_[0][0],)
#     for j in range(mix_num):
#         res = to_edge(model.means_[0][j][0], model.means_[0][j][1], 0)
#         new_means0.append(res)
#
#     for j in range(mix_num):
#         res = to_edge(model.means_[1][j][0], model.means_[1][j][1], 1)
#         new_means1.append(res)
#
#     for j in range(mix_num):
#         res = to_edge(model.means_[2][j][0], model.means_[2][j][1], 2)
#         new_means2.append(res)
#
#     for j in range(mix_num):
#         res = to_edge(model.means_[3][j][0], model.means_[3][j][1], 3)
#         new_means3.append(res)
#
#     for j in range(mix_num):
#         res = to_edge(model.means_[4][j][0], model.means_[4][j][1], 4)
#         new_means4.append(res)
#
#     for j in range(mix_num):
#         res = to_edge(model.means_[5][j][0], model.means_[5][j][1], 5)
#         new_means5.append(res)
#
#     # for j in range(mix_num):
#     #     res = to_edge(model.means_[6][j][0], model.means_[6][j][1], 6)
#     #     new_means6.append(res)
#
#     new_means = [new_means0, new_means1, new_means2, new_means3, new_means4, new_means5]# ,new_means6
#
#     new_sp = model.startprob_#以前的原封不动的
#     new_tm = model.transmat_#以前的原封不动的
#     new_cov = model.covars_#以前的原封不动的
#     new_w = model.weights_#以前的原封不动的
#
#     new_means = np.array(new_means)#经过校正的
#     print("new_means")
#     print(new_means)
#     print("new_cov")
#     print(new_cov)
#     print("new_w")
#     print(new_w)
#
#
#     model = GMMHMM(n_components=component_num, n_mix=mix_num, covariance_type='full', n_iter=iter_num, verbose=True,
#                    init_params='')  # 'stmcw'
#
#     model.startprob_ = new_sp
#     model.transmat_ = new_tm
#     model.covars_ = new_cov
#     model.weights_ = new_w
#     model.means_ = new_means #修正均值矩阵
#
#     print("end")
#     print("model.covars_")
#     print(model.covars_)
#     print("model.means_")
#     print(model.means_)
#     print("model.weights_")
#     print(model.weights_)
#
# #高斯混合输出图片
# for i in range(component_num):
#     for j in range(mix_num):
#         #x_y = (model.means_[i][0], model.means_[i][1])
#         cv2.circle(img, (int(model.means_[i][j][0]*width), int(model.means_[i][j][1]*height)), 5, (0, 0, 255), -1)
#         cv2.putText(img, str(i)+str(j), (int(model.means_[i][j][0]*width), int(model.means_[i][j][1]*height)), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)
# cv2.imwrite(target_file1, img)
#
#
# # # 单高斯输出图片
# # for i in range(component_num):
# #     #x_y = (model.means_[i][0], model.means_[i][1])
# #     cv2.circle(img, (int(model.means_[i][0]*width), int(model.means_[i][1]*height)), 5, (0, 0, 255), -1)
# #     cv2.putText(img, str(i), (int(model.means_[i][0]*width), int(model.means_[i][1]*height)), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)
# # cv2.imwrite(target_file1, img)
#
# # model.means_ = pre_means_all #赋值初始矩阵
# #
# # # print("model.means_")
# # # print(model.means_)
# # # print("pre_covs_ ")
# # # print(pre_covs_.shape)
# # # print(pre_covs_)
# # #model.covars_ = pre_covs_
# # # #shape==2 diag???? 负值??
# # #print(model.covars_.shape )
#
#
# # model.fit(X_all, lengths)#拟合函数
# # #
# # print("隐藏状态的个数", model.n_components)  #
# # print("均值矩阵")
# # print(model.means_)
# # print("协方差矩阵")
# # print(model.covars_)
# # print("状态转移矩阵--A")
# # print(model.transmat_)
# #
# #
# # #拟合观测序列
# # O_seq = np.array(X_sum[0])
# # print("O_seq")
# # print(O_seq)
# # state_sequence = model.predict(O_seq, lengths=None)
# # print("state_sequence")
# # print(state_sequence) #预测最可能的隐藏状态
# #
# # np.savetxt('out/state_sequence.txt', state_sequence, fmt="%.3f", delimiter=',') #保存为3位小数的浮点数，用逗号分隔
# #
# # print("predict_proba")
# # posterior_probability = model.predict_proba(O_seq, lengths=None)
# # print(posterior_probability)
# # np.savetxt('out/predict_proba.txt', posterior_probability, fmt="%.3f", delimiter=',') #保存为3位小数的浮点数，用逗号分隔
# #
# #
# # # print("predict_proba")
# # # pp = model.predict_proba(X_all)
# # # print(pp)
# # # print("model.monitor_.converged")
# # # print(model.monitor_.converged)
# # # # cv2.putText(img, str(i), (int(model.means_[i][0]), int(model.means_[i][1])), cv2.FONT_ITALIC, 0.9, (210, 50, 220),
# # # #             2, cv2.LINE_AA)
# #
#
# #
# # # 单高斯输出图片
# # for i in range(component_num):
# #         #x_y = (model.means_[i][0], model.means_[i][1])
# #         cv2.circle(img, (int(model.means_[i][0]*width), int(model.means_[i][1]*height)), 5, (0, 0, 255), -1)
# #         cv2.putText(img, str(i), (int(model.means_[i][0]*width), int(model.means_[i][1]*height)), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)
# #
# #
# # # #高斯混合输出图片
# # # for i in range(component_num):
# # #     for j in range(mix_num):
# # #         #x_y = (model.means_[i][0], model.means_[i][1])
# # #         cv2.circle(img, (int(model.means_[i][j][0]*width), int(model.means_[i][j][1]*height)), 5, (0, 0, 255), -1)
# # #         cv2.putText(img, str(i)+str(j), (int(model.means_[i][j][0]*width), int(model.means_[i][j][1]*height)), cv2.FONT_ITALIC, 0.9, (210, 50, 220), 2, cv2.LINE_AA)
# #
# # cv2.imwrite(target_file, img)
# #
#
# # #输出置信度是0.95的高斯混合模型，需要归一化
# # plt.rcParams["figure.figsize"] = (10.0, 10.0)
# # fig, ax = plt.subplots()
# # ax.set_xlabel("x")
# # ax.set_ylabel("y")
# # confidence = 5.991
# # color = "red"
# # alpha = 0.3
# # eigv = False
# # for i in range(component_num):
# #     make_ellipses(model.means_[i], model.covars_[i], ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)
# #
# # plt.savefig('out/gaussian_covariance_matrix.png')
# # #plt.savefig("/out/gaussian_covariance_matrix.png")
# # #plt.show()
#
# #输出置信度是0.95的高斯混合模型，需要归一化
# plt.rcParams["figure.figsize"] = (10.0, 10.0)
# fig, ax = plt.subplots()
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# confidence = 5.991
# color = "blue"
# alpha = 0.3
# eigv = True
# for i in range(component_num):
#     for j in range(mix_num):
#         make_ellipses(str(i)+str(j), model.means_[i][j], model.covars_[i][j], ax, confidence=confidence, color=color, alpha=alpha, eigv=eigv)
#
# plt.savefig(cov_file)
# plt.show()
#
#
# print("model.transmat_")
# print(model.transmat_)
# print("model.weights_")
# print(model.weights_)
#
# # state_sequence0 = model.predict(np.array(X_sum[0]), lengths=None)
# # print("state_sequence0")
# # print(state_sequence0) #预测第0个人最可能的隐藏状态
# # np.savetxt('out/state_sequence0.txt', state_sequence0, fmt="%.3f", delimiter=',') #保存为3位小数的浮点数，用逗号分隔
# #
# #
# # state_sequence1 = model.predict(np.array(X_sum[1]), lengths=None)
# # print("state_sequence1")
# # print(state_sequence1) #预测第1个人最可能的隐藏状态
# # np.savetxt('out/state_sequence1.txt', state_sequence1, fmt="%.3f", delimiter=',') #保存为3位小数的浮点数，用逗号分隔
#
# #按照timestamp,AOI的形式打印
# Sequence0 = model.predict(np.array(X_sum[0]), lengths=None)
# timestamp_AOI0 = np.column_stack([timestamp_sum[0], Sequence0])
# np.savetxt('out/state_sequence0.txt', timestamp_AOI0, fmt="%d", delimiter=',')
#
# Sequence1 = model.predict(np.array(X_sum[1]), lengths=None)
# timestamp_AOI1 = np.column_stack([timestamp_sum[1], Sequence1])
# np.savetxt('out/state_sequence1.txt', timestamp_AOI1, fmt="%d", delimiter=',')
#
# Sequence2 = model.predict(np.array(X_sum[2]), lengths=None)
# timestamp_AOI2 = np.column_stack([timestamp_sum[2], Sequence2])
# np.savetxt('out/state_sequence2.txt', timestamp_AOI2, fmt="%d", delimiter=',')
#
# Sequence3 = model.predict(np.array(X_sum[3]), lengths=None)
# timestamp_AOI3 = np.column_stack([timestamp_sum[3], Sequence3])
# np.savetxt('out/state_sequence3.txt', timestamp_AOI3, fmt="%d", delimiter=',')
#
# Sequence4 = model.predict(np.array(X_sum[4]), lengths=None)
# timestamp_AOI4 = np.column_stack([timestamp_sum[4], Sequence4])
# np.savetxt('out/state_sequence4.txt', timestamp_AOI4, fmt="%d", delimiter=',')
#
# Sequence5 = model.predict(np.array(X_sum[5]), lengths=None)
# timestamp_AOI5 = np.column_stack([timestamp_sum[5], Sequence5])
# np.savetxt('out/state_sequence5.txt', timestamp_AOI5, fmt="%d", delimiter=',')
#
# Sequence6 = model.predict(np.array(X_sum[6]), lengths=None)
# timestamp_AOI6 = np.column_stack([timestamp_sum[6], Sequence6])
# np.savetxt('out/state_sequence6.txt', timestamp_AOI6, fmt="%d", delimiter=',')
#
# Sequence7 = model.predict(np.array(X_sum[7]), lengths=None)
# timestamp_AOI7 = np.column_stack([timestamp_sum[7], Sequence7])
# np.savetxt('out/state_sequence7.txt', timestamp_AOI7, fmt="%d", delimiter=',')
#
# Sequence8 = model.predict(np.array(X_sum[8]), lengths=None)
# timestamp_AOI8 = np.column_stack([timestamp_sum[8], Sequence8])
# np.savetxt('out/state_sequence8.txt', timestamp_AOI8, fmt="%d", delimiter=',')
#
# Sequence9 = model.predict(np.array(X_sum[9]), lengths=None)
# timestamp_AOI9 = np.column_stack([timestamp_sum[9], Sequence9])
# np.savetxt('out/state_sequence9.txt', timestamp_AOI9, fmt="%d", delimiter=',')
#
# Sequence10 = model.predict(np.array(X_sum[10]), lengths=None)
# timestamp_AOI10 = np.column_stack([timestamp_sum[10], Sequence10])
# np.savetxt('out/state_sequence10.txt', timestamp_AOI10, fmt="%d", delimiter=',')
#
# Sequence11 = model.predict(np.array(X_sum[11]), lengths=None)
# timestamp_AOI11 = np.column_stack([timestamp_sum[11], Sequence11])
# np.savetxt('out/state_sequence11.txt', timestamp_AOI11, fmt="%d", delimiter=',')
#
#
# output_hal = open(modle_file, 'wb')
# str = pickle.dumps(model)
# output_hal.write(str)
# output_hal.close()