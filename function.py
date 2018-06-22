# -*- coding: utf-8 -*-
"""
Created on Wed May 10 10:07:44 2017

@author: zangz
"""
import numpy as np
import matplotlib.pyplot as plt
import random as rd
from scipy.special import comb, perm


# ==============================================================================
# 单个多项式网络训练函数
# 输入： input1为输入的两行数据，每一行代表一个输入组，aim为目标的数据，训练的目的是
# 到最佳的w1,w2,使得w1*第一行数据加上w2乘以第二行数据与输出尽可能相似。
# 测试函数如下：
# input1 = np.array([[2,5,8,3,6,9,10,54,5],
#                    [9,8,34,21,6,9,2,4,1 ]])
# aim = np.array([2,5,8,3,6,9,10,54,5])
#
# [ w , output , i , errorn ]=Separatetraining(input1,aim)
#
# print(i,errorn)
# plt.figure(1)
# plt.plot(aim)
# plt.plot(output)#如果两条曲线相似，则说明训练成功
# ==============================================================================
def Separatetraining(input1, aim):
    # inputshape = input1.shape
    # aimshape = aim.shape

    # 随机生成初始矩阵
    w = np.array([rd.random(), rd.random()]) * 2 - 1
    errorn = 1000  # 初始误差
    errorlast = 1010  # 初始上次误差
    i = 0
    step = 0.5  # 步长
    while errorn < errorlast:
        errorlast = errorn
        errorlist = (np.mat(input1).T * np.mat(w).T).T - aim  # 计算误差矩阵
        abserrorlist = np.array(abs(errorlist))  # 计算误差矩阵的绝对值
        errorn = abserrorlist.sum()  # 计算误差和

        minput = np.mat(input1)  # 将输入参数矩阵化以便进行矩阵乘法
        # 利用widrow-hoff学习法则进行训练
        w = w - step * (errorlist * minput.T) / (input1 ** 2).sum()
        i = i + 1
        if i > 1000:
            break
        npw = np.array(w)
    output = np.array((np.mat(input1).T * np.mat(w).T).T)

    #    plt.figure()
    #    plt.plot(aim,'r')
    #    plt.plot(output,'k')
    #    #a = input()

    return [npw, output, i, errorn]


# ==============================================================================
# 对之前训练好的网络进行检验的函数，计算出其与检验数据的误差，用于挑选优秀的网络
# 输入：
#         w之前训练好的权值矩阵
#         sepratetextin检验用的输入数据
#         sepratetextout检验用的输出数据
# 输出：
#         errorn得到的检验误差
# ==============================================================================
def Getsuitlevel(w, sepratetextin, sepratetextout):
    output = np.array((np.mat(sepratetextin).T * np.mat(w).T).T)
    errorlist = (np.mat(sepratetextin).T * np.mat(w).T).T - sepratetextout  # 计算误差矩阵
    abserrorlist = np.array(abs(errorlist))  # 计算误差矩阵的绝对值
    errorn = [abserrorlist.sum()]  # 计算误差和
    return [errorn, output]


# ==============================================================================
# 用单个多项式训练函数与单个的检验韩式对输入数据进行成组训练的函数并成组检验的函数
# 输入：GMDHinputrain是训练用的输入数据，n行数据，
#       GMDHoutputrain是训练用做目标的1行数据
#       GMDHinputtext是检验用的输入数据，n行数据，
#       GMDHoutputtext是检验用做目标的1行数据
# 输出：训练好的w的组，每行两个数据分别是权值w1,与w2，和检验好的误差，
#       将在函数外进行有效的选择
# 测试函数如下：
# data = np.random.random(size=(16,18))
# #所有数据归一
# data = (data - data.min())/data.max()
# GMDH = []
# GMDHinputnum = 15# 十五组数据输入
# GMDHoutputnum = 1# 一组数据输出
# GMDHtrainnum = 9
# GMDHtextnum = 8
# GMDHpredictnum = 1
# GNDHcomb = comb(GMDHtrainnum,2)
# maxgeneration = 50
# GMDHinputrain = data[0:GMDHinputnum , 0 :GMDHtrainnum ]
# GMDHinputtext = data[0:GMDHinputnum , GMDHtrainnum :GMDHtrainnum+GMDHtextnum ]
# GMDHinputpredict = data[0:GMDHinputnum , GMDHtrainnum+GMDHtextnum :GMDHtrainnum+GMDHtextnum+GMDHpredictnum ]
# GMDHoutputrain = data[GMDHinputnum:GMDHoutputnum+GMDHinputnum , 0 :GMDHtrainnum ]
# GMDHoutputtext = data[GMDHinputnum:GMDHoutputnum+GMDHinputnum, GMDHtrainnum :GMDHtrainnum+GMDHtextnum ]
# GMDHoutputpredict = data[GMDHinputnum:GMDHoutputnum+GMDHinputnum , GMDHtrainnum+GMDHtextnum :GMDHtrainnum+GMDHtextnum+GMDHpredictnum ]
# wcollection = Grouptraining(GMDHinputrain,GMDHoutputrain,GMDHinputtext,GMDHoutputtext)
# ==============================================================================
def Grouptraining(GMDHinputrain, GMDHoutputrain, GMDHinputtext, GMDHoutputtext):
    inputtrainshape = GMDHinputrain.shape  # 确定输入的数组的行数和列数
    # outputtrainshape = GMDHoutputrain.shape

    inputtextshape = GMDHinputtext.shape  # 确定输入的数组的行数和列数
    # outputtextshape = GMDHoutputtext.shape

    GMDHtrainnum = inputtrainshape[0]  # 得到用来训练的数据个数
    GMDHtextnum = inputtextshape[0]

    sepratetrain = np.zeros((2, inputtrainshape[1]))  # 初始化Separatetraining函数的输入矩阵
    sepratetrainout = np.zeros((1, inputtrainshape[1]))  # 初始化Separatetraining函数的输出矩阵

    sepratetextin = np.zeros((2, inputtextshape[1]))  # 初始化Separatetraining函数的输入矩阵
    sepratetextout = np.zeros((1, inputtextshape[1]))  # 初始化Separatetraining函数的输出矩阵

    corrantn = 0
    GNDHcomb = comb(GMDHtrainnum, 2)
    trainoutput = np.zeros((GNDHcomb, inputtrainshape[1]))
    textoutput = np.zeros((GNDHcomb, inputtextshape[1]))
    wcollection = np.zeros((GNDHcomb, 5))  # 初始话将要输出的权值矩阵
    for i in range(GMDHtrainnum):  # 用两重循环遍历所有的两两组合
        for j in range(i + 1, GMDHtrainnum):
            # print(i,j)
            sepratetrain[0, :] = GMDHinputrain[i, :]  # 装填输入矩阵
            sepratetrain[1, :] = GMDHinputrain[j, :]
            sepratetrainout[0, :] = GMDHoutputrain[0, :]  # 装填输出矩阵

            sepratetextin[0, :] = GMDHinputtext[i, :]  # 装填输入矩阵
            sepratetextin[1, :] = GMDHinputtext[j, :]
            sepratetextout[0, :] = GMDHoutputtext[0, :]  # 装填输出矩阵

            [w, output1, loopnum, errorn] = Separatetraining(sepratetrain, sepratetrainout)  # 调用计算
            trainoutput[corrantn, :] = output1[0, :]
            if (i == 0 and j == 2):
                plt.figure(99)
                plt.plot(output1[0, :], 'r')
                plt.plot(sepratetrainout[0, :])
            # a = input()

            [error, output2] = Getsuitlevel(w, sepratetextin, sepratetextout)
            textoutput[corrantn, :] = output2

            w1w2errorij = [w[0, 0], w[0, 1], error[0], i, j]
            wcollection[corrantn, :] = w1w2errorij  # 保存权值
            corrantn = corrantn + 1
    return [wcollection, trainoutput, textoutput]

# data = np.random.random(size=(16,18))
##所有数据归一
# data = (data - data.min())/data.max()
# GMDH = []
# GMDHinputnum = 15# 十五组数据输入
# GMDHoutputnum = 1# 一组数据输出
# GMDHtrainnum = 9
# GMDHtextnum = 8
# GMDHpredictnum = 1
# GNDHcomb = comb(GMDHtrainnum,2)
# maxgeneration = 50
# GMDHinputrain = data[0:GMDHinputnum , 0 :GMDHtrainnum ]
# GMDHinputtext = data[0:GMDHinputnum , GMDHtrainnum :GMDHtrainnum+GMDHtextnum ]
# GMDHinputpredict = data[0:GMDHinputnum , GMDHtrainnum+GMDHtextnum :GMDHtrainnum+GMDHtextnum+GMDHpredictnum ]
# GMDHoutputrain = data[GMDHinputnum:GMDHoutputnum+GMDHinputnum , 0 :GMDHtrainnum ]
# GMDHoutputtext = data[GMDHinputnum:GMDHoutputnum+GMDHinputnum, GMDHtrainnum :GMDHtrainnum+GMDHtextnum ]
# GMDHoutputpredict = data[GMDHinputnum:GMDHoutputnum+GMDHinputnum , GMDHtrainnum+GMDHtextnum :GMDHtrainnum+GMDHtextnum+GMDHpredictnum ]
# wcollection = Grouptraining(GMDHinputrain,GMDHoutputrain,GMDHinputtext,GMDHoutputtext)

























