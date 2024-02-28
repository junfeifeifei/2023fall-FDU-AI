import numpy as np
from sklearn_crfsuite import CRF
from util import *


class CRFModel(object):
    def __init__(self,
                 type,
                 algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False
                 ):

        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)
        self.type = type # 1代表英语，2代表汉语

    def train(self, sentences, tag_lists):
        features = [sent2features(s,self.type) for s in sentences]
        self.model.fit(features, tag_lists)

    def test(self, testWordLists, testTagLists, wordDict, tagDict,type):
        features = [sent2features(s,self.type) for s in testWordLists]
        tagPres = self.model.predict(features)
        testTagLists = np.hstack(testTagLists)
        tagPres_arr = np.hstack(tagPres)
        correct = np.sum(testTagLists == tagPres_arr)
        accuracy = correct/len(testTagLists)
        print(accuracy)

        predict = tagPres
        wordList = testWordLists
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

