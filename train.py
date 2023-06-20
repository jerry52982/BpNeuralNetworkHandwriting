from NeuralNetwork import NeuralNetwork
import numpy as np
import pickle
import csv

def train():
    file_name = 'C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data'	# 数据集为60000张带标签的28x28手写数字图像
    y = []
    x = []
    y_t = []
    x_t = []
    with open("C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\训练数据整合.csv", 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader) #遍历csv文件的每一行
        print(header_row)
        for row in reader:
            if np.random.random() < 0.8:	#大约80%的数据用于训练
                y.append(int(row[0]))#第一列为待识别对象的正确结果，将正确结果添加到y列表中
                x.append(list(map(int, row[1:])))#第1位开始为图片转化成的矩阵，将矩阵添加到x列表中
            else:#剩余的图片正确结果以及矩阵分别添加到y_t和x_t列表中用来测试
                y_t.append(int(row[0]))
                x_t.append(list(map(int, row[1:])))
    len_train = len(y)
    len_test = len(y_t)
    print('训练集大小%d，测试集大小%d' % (len_train, len_test))
    x = np.array(x)
    y = np.array(y)
    nn = NeuralNetwork([784, 784, 10])	# 神经网络各层神经元个数
    nn.fit(x, y)
    file = open('C:\Users\韦祖垚\Desktop\模式识别大作业韦祖垚\data\\NN.txt', 'wb')
    pickle.dump(nn, file) #将nn文件转换为python可以识别的数据对象
    count = 0
    for i in range(len_test):
        p, _ = nn.predict(x_t[i]) #将预测结果赋给p
        if p == y_t[i]:
            count += 1 #如果预测结果和正确结果一样，count加一
    print('模型识别正确率：', count/len_test)

    for i in range(10): #遍历0~9十个数字
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for j in range(len_test): #遍历测试集所有行
            p, _ = nn.predict(x_t[j])
            if (p == y_t[j] and p==i):
                TP+=1
            elif(p == y_t[j] and p!=i):
                TN+=1
            elif(p!=y_t[j] and p==i):
                FP+=1
            elif(p!=y_t[j] and p!=i):
                FN+=1
        print('数字',i,'的准确率:',(TP+TN)/(TP+FP+FN+TN))
        print('数字',i,'的精确率:',(TP) / (TP + FP ))
        print('数字',i,'的召回率:',(TP) / (TP + FN))






def mini_test():	# 测试神经网络能正常运行
    x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 2, 3]
    nn = NeuralNetwork([2, 4, 16, 4])
    nn.fit(x, y, epochs=10000)
    for i in x:
        print(nn.predict(i))


# mini_test()
train()
