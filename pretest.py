import os
import csv
import pandas as pd
def all_csv1(file_PATH = r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data',save_file="测试数据整合.csv"):
    df_list = [] #创建新列表用来存储提取出来的列表
    df = pd.read_csv(r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\test_labels.csv',header=None)# 读取CSV文件数据
    df.columns = ['labels']
    data = df.iloc[:,:]# 选取文件中某行某列数据
    df_list.append(data)# 将选取的数据添加到列表
    df = pd.read_csv(r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\test_img.csv',header=None)
    df.columns = [x for x in range (1,785)] # 添加自定义的columns的名字
    data = df.iloc[:,:]
    df_list.append(data)
    df2 = pd.concat(df_list,axis=1)#将列表数据按列合并，axis=1表示按列整合
    open('C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\测试数据整合.csv', 'wb')
    df2.to_csv(r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\测试数据整合.csv',index=False)#将整合好的数据输入到新建的csv文件中

def all_csv2(file_PATH = r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data',save_file="训练数据整合.csv"):
    df_list = []
    df = pd.read_csv(r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\train_labels.csv',header=None)
    df.columns = ['labels']
    data = df.iloc[:,:]
    df_list.append(data)
    df = pd.read_csv(r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\train_img.csv',header=None)
    df.columns = [x for x in range (1,785)]
    data = df.iloc[:,:]
    df_list.append(data)
    df2 = pd.concat(df_list,axis=1)
    open('C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\训练数据整合.csv', 'wb')
    df2.to_csv(r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\训练数据整合.csv',index=False)

all_csv1(file_PATH = r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data',save_file="测试数据整合.csv")
all_csv2(file_PATH = r'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data',save_file="训练数据整合.csv")