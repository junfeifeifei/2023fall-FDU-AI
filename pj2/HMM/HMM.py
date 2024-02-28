import numpy
import numpy as np
from util import *

class HMM():
    '''
    隐马尔可夫模型 (HMM) 类
    wordDictSize: 词表长度
    tagDictSize: 标签字典长度
    emitProb: 观测概率矩阵
    transitionProb: 转移概率矩阵
    initProb: 初始概率向量
    '''

    def __init__(self, wordDictSize, tagDictSize):
        self.wordDictSize = wordDictSize
        self.tagDictSize = tagDictSize
        self.transitionProb = np.random.rand(self.tagDictSize, self.tagDictSize)
        for index in range(self.tagDictSize):
            self.transitionProb[index] = self.transitionProb[index] / np.sum(self.transitionProb[index])

        self.initProb = numpy.random.rand(self.tagDictSize)
        self.initProb = self.initProb / np.sum(self.initProb)

        self.emitProb = numpy.random.rand(self.tagDictSize, self.wordDictSize)
        for index in range(self.tagDictSize):
            self.emitProb[index] = self.emitProb[index] / np.sum(self.emitProb[index])

    def trainSup(self, trainWordLists, trainTagLists):
        '''
        监督训练HMM模型

        参数：
        - trainWordLists: 训练数据中的词列表
        - trainTagLists: 训练数据中的标签列表

        该函数根据给定的训练数据，计算并更新HMM模型的转移概率、发射概率和初始概率。
        '''

        # 初始化概率矩阵
        self.transitionProb = numpy.zeros((self.tagDictSize, self.tagDictSize))
        self.initProb = numpy.zeros(self.tagDictSize)
        self.emitProb = numpy.zeros((self.tagDictSize, self.wordDictSize))

        #遍历训练数据
        for i in range(len(trainWordLists)):
            for j in range(len(trainWordLists[i])):
                word, tag = trainWordLists[i][j], trainTagLists[i][j]
                # 更新初始概率和发射概率
                self.initProb[tag] += 1
                self.emitProb[tag][word] += 1
                # 如果不是最后一个词，则更新转移概率
                if j < len(trainWordLists[i]) - 1:
                    nextTag = trainTagLists[i][j + 1]
                    self.transitionProb[tag][nextTag] += 1

        # 归一化概率矩阵
        self.initProb = self.initProb / (self.initProb.sum())
        for index, value in enumerate(self.emitProb.sum(axis=1)):
            if value == 0: continue
            self.emitProb[index, :] = self.emitProb[index, :] / value

        for index, value in enumerate(self.transitionProb.sum(axis=1)):
            if value == 0: continue
            self.transitionProb[index, :] = self.transitionProb[index, :] / value

        # 防止概率为0，取一个很小的数代替
        self.initProb[self.initProb == 0] = 1e-10
        self.transitionProb[self.transitionProb == 0] = 1e-10
        self.emitProb[self.emitProb == 0] = 1e-10


    def viterbiAlg(self, sentence):
        '''
        维特比算法

        参数：
        - sentence: 输入句子的词序列

        返回：
        - state: 最优路径的状态序列
        '''

        sentenceSize = len(sentence)
        # score: 一个二维数组，其中score[i][j]表示在观测到第i个词的情况下，以状态j结尾的最大概率路径的概率值
        score = numpy.zeros((sentenceSize, self.tagDictSize))
        # 一个二维数组，其中path[i][j]表示在观测到第i个词的情况下，以状态j结尾的最大概率路径的上一个状态。
        path = numpy.zeros((sentenceSize, self.tagDictSize))
        score[0] = self.initProb + self.emitProb[:, sentence[0]]
        state = numpy.zeros(sentenceSize)

        # 递推计算Viterbi路径的分数和路径
        for index, word in enumerate(sentence):
            if index == 0: continue
            temp = score[index - 1] + self.transitionProb.T
            path[index] = numpy.argmax(temp, axis=1)
            score[index] = [element[int(path[index, i])] for i, element in enumerate(temp)] + self.emitProb[:, word]

        state[-1] = numpy.argmax(score[-1])

        for i in reversed(range(sentenceSize)):
            if i == sentenceSize - 1: continue
            state[i] = path[i + 1][int(state[i + 1])]
        return state

    def test(self, testWordLists, testTagLists, wordDict, tagDict, wordList, type):
        '''
        在测试集上评估模型性能，并输出结果

        参数：
        - testWordLists: 测试数据中的词列表
        - testTagLists: 测试数据中的标签列表
        - wordDict: 词典
        - tagDict: 标签典
        - wordList: 词列表
        - type: 类型，1表示英文，其他表示中文

        输出：
        - 输出模型在测试集上的准确率，并将预测结果写入文件
        '''

        # 将概率矩阵取对数
        self.transitionProb = numpy.log10(self.transitionProb)
        self.emitProb = numpy.log10(self.emitProb)
        self.initProb = numpy.log10(self.initProb)

        real, predict = [], []

        # 对测试集进行预测
        for sentence, tag in zip(testWordLists, testTagLists):
            tagPre = self.viterbiAlg(sentence)
            real.append(int2str(tag, tagDict))
            predict.append(int2str(tagPre, tagDict))

        # 计算准确率
        real = np.hstack(real)
        predict_arr = np.hstack(predict)
        correct = np.sum(real == predict_arr)
        accuracy = correct / len(real)
        print(accuracy)

        length = len(predict)
        if type == 1:
            with open('output_English.txt', 'w') as file:
                for i in range(length):
                    word = wordList[i]
                    col = len(word)
                    for j in range(col):
                        file.write(wordList[i][j])
                        file.write(' ')
                        file.write(predict[i][j])
                        file.write('\n')
                    file.write('\n')
        else:
            with open('output_Chinese.txt', 'w') as file:
                for i in range(length):
                    word = wordList[i]
                    col = len(word)
                    for j in range(col):
                        file.write(wordList[i][j])
                        file.write(' ')
                        file.write(predict[i][j])
                        file.write('\n')
                    file.write('\n')
