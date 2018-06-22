# ==============================================================================
# # -*- coding: utf-8 -*-
# ==============================================================================
"""
Created on Tue May  9 18:47:56 2017
本程序使用GMDH网络对交通的流量进行预测，输入的数据为连续n天的m组流量数据。
输出数据为第n+1天的m组的流量的预测数据。
每个神经元的学习方式为widrow-hoff
在学习过程中，每一层否挑选15个优秀的神经元保留到下一层
@author: jon zang，杭州，15168307480，zangzelin@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.special import comb, perm
import function as ft
import copy
import winsound

# ==============================================================================
# 方法的作用：
# ￼￼本方法使用GMDH算法（多项式神经网络）对交通的流量做预测
# 方法的输入输出：
# 输入的数据为连续n天的不同时间段内流量数据（一共m（m=17）个时间段）。
# 输出数据为第n+1天(下一天)的m组的流量的预测数据。
# 数据的分配
# 分配九天的数据作为训练，八天的数据作为测试
# 网络的结构：
# 使用如下结构对网络进行训练
# 输入层：9个输入层，每个输入层对应一天的车流量数据
# 隐含层1：对9个输入两两组合得到36种组合，分别对每种组合使用widrow-hoff法则进行训练，
# 后通过测试数据选取误差最小的15个神经元保留作为隐含层1的神经元。
# 隐含层2：对15个输入两两组合得到105种组合，分别对每种组合使用widrow-hoff法则进行训
# 练，后通过测试数据选取误差最小的15个神经元保留作为隐含层1的神经元。
# 隐含层的层数按照上述法则进行训练，直到网络的误差不再下降，出现上升。后停止网络，寻
# 找最后一层最好的一个神经元作为输出。
# ==============================================================================

'''
程序参数设置
'''
interval = 1
GMDHinputnum = 50  # 十五组数据输入
GMDHoutputnum = 1  # 一组数据输出
GMDHtrainnum = 9*16  # 用于训练的数据数量
GMDHtextnum = 8*16  # 用于测试的数据数量
GMDHpredictnum = 1  # 用于预测的数据数量
GNDHcomb = comb(GMDHinputnum, 2)  # 训练数据的组合数，用于训练数据的两两组合
maxgeneration = 50 # 算法最大代数
num_of_group_keep_to_next_layer = GMDHinputnum  # 算法挑选用来作为下一层输入的

num_of_start_study_group = 0  # 选择学士所用的数据从那一组开始

num_of_end_study_group = num_of_start_study_group + GMDHinputnum + GMDHoutputnum  # 选择学习终止的数据

'''
导入数据
'''
# 导入数据
data_oneyear = np.loadtxt('data_oneyear.csv', dtype=np.float, delimiter=',')  # 读入数据
datamin = data_oneyear.min()  # 计算最小值
datamax = data_oneyear.max()  # 计算最大值
data_normalization = (data_oneyear - datamin) / datamax  # 利用最大值最小值进行归一化
Number_of_data_Group = data_normalization.shape[1]  # 计算输入数据的数据组数
Number_of_each_Group_data = data_normalization.shape[0]  # 计算输入数据的每组的个数
print('输入数据的组数为', Number_of_data_Group)
print('每组数据的个数为', Number_of_each_Group_data)
print('使用数据的组数', GMDHinputnum)
print('开始于', num_of_start_study_group, '结束于', num_of_end_study_group)
'''
数据的划分
'''
data = data_normalization[0:Number_of_each_Group_data:interval,
       num_of_start_study_group: num_of_end_study_group].transpose()  # 数据划分，将有用的数据进行提取
print(data.shape)
# 输入
GMDHinputrain = data[0:GMDHinputnum, 0:GMDHtrainnum]
GMDHinputtext = data[0:GMDHinputnum, GMDHtrainnum:GMDHtrainnum + GMDHtextnum]
GMDHinputpredict = data[0:GMDHinputnum, GMDHtrainnum + GMDHtextnum:GMDHtrainnum + GMDHtextnum + GMDHpredictnum]

# 输出
GMDHoutputrain = data[GMDHinputnum:GMDHoutputnum + GMDHinputnum, 0:GMDHtrainnum]
GMDHoutputtext = data[GMDHinputnum:GMDHoutputnum + GMDHinputnum, GMDHtrainnum:GMDHtrainnum + GMDHtextnum]
GMDHoutputpredict = data[GMDHinputnum:GMDHoutputnum + GMDHinputnum,
                    GMDHtrainnum + GMDHtextnum:GMDHtrainnum + GMDHtextnum + GMDHpredictnum]
'''
网络的构建
每层网络设定为y=ax1+bx2的形式
'''
# 网络缓存参数
bestfit = 100000  # 最优准则值，初始值1000（任意的）
lastbestfit = 100001  # 上一次的最优准则值，初始值1001（任意的）
currentlayer = 0  # 当前第0层
allcoefficient = np.zeros((GNDHcomb, 5 * maxgeneration))  # 初始化权值误差索引矩阵
allerror = np.zeros((maxgeneration, int(GNDHcomb), 2))  # 初始化误差矩阵
bestoutput = np.zeros((num_of_group_keep_to_next_layer, GMDHtrainnum))  # 初始化最佳输出矩阵
mid = np.zeros((1, 5))  # 初始化交换用的中转矩阵
errorhistory = np.zeros((maxgeneration, 1))  # 初始化历史误差矩阵
# 主循环
newinput = GMDHinputrain  # 给训练输入矩阵装入初值，并初始化
newoutput = GMDHinputtext  # 给训练输出矩阵装入初值，并初始化
newinputtext = GMDHinputtext  # 给测试输入矩阵装入初值，并初始化
errorlist_in_study = np.zeros( ( maxgeneration ) )  # 初始化迭代误差清单

while not (lastbestfit < bestfit and currentlayer > 4):
    # 使用训练数据，计算当前层的每一个多项式y=ax1+bx2中的a,b
    [wcollection, trainoutput, textoutput] = ft.Grouptraining(newinput, GMDHoutputrain, newoutput, GMDHoutputtext)
    # 使用验证数据计算误差
    if currentlayer == 0:  # 调试用，无实际意义
        tiaoshi1 = copy.deepcopy(trainoutput)
        tiaoshi2 = copy.deepcopy(textoutput)

    allcoefficient[:, currentlayer * 5:currentlayer * 5 + 5] = wcollection  # 保存计算所得权值、误差、检索矩阵
    errorse = allcoefficient[:, 2 + currentlayer * 5]  # 将权值、误差、检索矩阵中的误差提取出来
    errorsum = errorse.sum()  # 误差进行加和
    lastbestfit = bestfit  # 保存上次的适应度
    bestfit = errorsum  # 计算总的适应度
    errorlist_in_study[currentlayer] = bestfit
    errorhistory[currentlayer] = bestfit
    # 使用冒泡排序寻找匹配度最高的15个组合
    for i in range(int(GNDHcomb)):
        for j in range(int(GNDHcomb) - 1):
            if allcoefficient[j, 5 * currentlayer + 2] > allcoefficient[j + 1, 5 * currentlayer + 2]:
                # 交换权值、误差、检索矩阵
                mid1 = copy.deepcopy(allcoefficient[j, 5 * currentlayer:currentlayer * 5 + 5])
                allcoefficient[j, 5 * currentlayer:currentlayer * 5 + 5] = allcoefficient[j + 1,
                                                                           5 * currentlayer:currentlayer * 5 + 5]
                allcoefficient[j + 1, 5 * currentlayer:currentlayer * 5 + 5] = mid1
                # 交换训练输出值矩阵
                mid2 = copy.deepcopy(trainoutput[j, :])
                trainoutput[j, :] = trainoutput[j + 1, :]
                trainoutput[j + 1, :] = mid2
                # 交换测试输出值矩阵
                mid3 = copy.deepcopy(textoutput[j, :])
                textoutput[j, :] = textoutput[j + 1, :]
                textoutput[j + 1, :] = mid3

    # 使用训练数据，计算被选中的输出
    newinput = trainoutput[0:num_of_group_keep_to_next_layer, :]
    newoutput = textoutput[0:num_of_group_keep_to_next_layer, :]

    currentinput = bestoutput  # 15*9
    # 终止条件
    currentlayer = currentlayer + 1
    print('第', currentlayer, '层学习，最大层数', maxgeneration)
    if currentlayer > maxgeneration - 1:
        break
abovelayernumber = currentlayer


np.savetxt("textoutput.csv", textoutput, fmt="%f", delimiter=",")
np.savetxt("trainoutput.csv", trainoutput, fmt="%f", delimiter=",")
np.savetxt("allcoefficient.csv", allcoefficient, fmt="%f", delimiter=",")
np.savetxt("dataout.csv", data, fmt="%f", delimiter=",")
# ==============================================================================
# 绘制学习图像
# ==============================================================================

# 构建预测矩阵和真实数据矩阵
output = np.zeros((1, newinput.shape[1]+newoutput.shape[1]))
output[0, 0:newinput.shape[1]] = newinput[0, :]
output[0, newinput.shape[1]:newinput.shape[1]+newoutput.shape[1]] = newoutput[0, :]
outputuse = output[0, 1:newinput.shape[1]+newoutput.shape[1]-1]
# 真实
real = np.zeros(newinput.shape[1]+newoutput.shape[1])
a = newinput.shape[1]+newoutput.shape[1]
real[:] = data[GMDHinputnum, 0:a]
realuse = real[1:newinput.shape[1]+newoutput.shape[1]-1]

# 计算误差
errorpredict = realuse - outputuse  # 误差向量
errorpredictsum1 = errorpredict.sum()  # 平均误差分子
sum11 = realuse.sum()  # 误差分母
aberror = abs(errorpredict)  # 误差求绝对值
errorpredictsum2 = aberror.sum()  # 绝对误差分子
# 打印误差
print(errorpredictsum1 / sum11)
print(errorpredictsum2 / sum11)

plt.figure(1)
plt.plot(outputuse[:] * datamax + datamin, 'r', label="predict")
plt.plot(realuse[:] * datamax + datamin, label="data")
plt.legend(loc='upper right')
plt.xlabel("time(5min)")
plt.ylabel("followrate(car)")
plt.text(100, 690.8, 'average error = 0.4% \n absolute error = 2%', color='black', ha='center')
#plt.show()
ax = plt.plot(outputuse[:] * datamax + datamin, 'r', label="predict")
# ==============================================================================
# 绘制预测图像
# 通过新输入的数据对网络的预测能力进行衡量
# ==============================================================================
# 导入叫大数据量的数组

num_of_text_time = 50
errorlist_in_test = np.zeros((num_of_text_time))
sueshuju = 2  # 选择使用的数据组数，选择0为使用学习的数据，每多加1，数据向后挫一位
# 建立数据处理矩阵，将288组数据压缩成16组
for sueshuju in range(num_of_text_time):

    pdata = data_normalization[0:Number_of_each_Group_data:interval,\
            num_of_start_study_group + sueshuju: num_of_end_study_group + sueshuju].transpose()


    predictdata = pdata[0:GMDHinputnum, :]  # 选取输入数据
    # predictoutput  = np.zeros()
    comefrom = np.zeros((2))  # 网络结构检索号保存矩阵初始化
    pcurrentinput = predictdata[0:num_of_group_keep_to_next_layer, :]  # 为当前的矩阵赋初值
    pinput = np.zeros((2, Number_of_each_Group_data))  # 初始化输入矩阵
    poutput = np.zeros((num_of_group_keep_to_next_layer, Number_of_each_Group_data))  # 初始化输出矩阵
    pw = np.zeros(2)  # 初始化权值矩阵

    # 遍历网络层
    for i in range(abovelayernumber):
        # 遍历选中的组合
        for j in range(GMDHinputnum):
            # 取出权值
            pw[0] = allcoefficient[j, i * 5 + 0]
            pw[1] = allcoefficient[j, i * 5 + 1]
            # 取出连接的输入
            comefrom[0] = allcoefficient[j, i * 5 + 3]
            comefrom[1] = allcoefficient[j, i * 5 + 4]
            # 取出需要用的两行输入
            pinput[0, :] = pcurrentinput[comefrom[0], :]
            pinput[1, :] = pcurrentinput[comefrom[1], :]

            poutput[j] = pinput[0, :] * pw[0] + pinput[1, :] * pw[1]
        pcurrentinput = poutput

    # 绘图分析

    errortest = (poutput[0, :]-pdata[GMDHinputnum, :])
    errortestsum = abs(poutput[0, :]).sum()
    errorlist_in_test[sueshuju] = abs(errortest).sum()/errortestsum

    plt.figure(2+sueshuju)
    plt.plot(poutput[0, :] * datamax + datamin, 'r', label="predict")
    plt.plot(pdata[GMDHinputnum, :] * datamax + datamin, label="data")
    plt.legend(loc='upper right')
    plt.xlabel("time(80min)")
    plt.ylabel("followrate(car)")
    plt.text(8.5, 8990.8, 'average error = 1% \n absolute error = 2%', color='black', ha='center')


winsound.Beep(367, 1000)
print('平均误差', errorlist_in_test.sum()/num_of_text_time)
plt.show()












