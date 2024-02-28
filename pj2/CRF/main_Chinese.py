from CRF import *
from check import check
wordDict = acquireDict(['../NER/Chinese/train.txt'])
tagDict = {'O': 0,
           'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 'S-NAME': 4,
           'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8,
           'B-EDU': 9, 'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12,
           'B-TITLE': 13, 'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16,
           'B-ORG': 17, 'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20,
           'B-RACE': 21, 'M-RACE': 22, 'E-RACE': 23, 'S-RACE': 24,
           'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 'S-PRO': 28,
           'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32}


trainWordLists, trainTagLists = prepareData('../NER/Chinese/train.txt')
testWordLists, testTagLists = prepareData('../NER/Chinese/chinese_test.txt')

crf = CRFModel(2)
crf.train(trainWordLists, trainTagLists)
crf.test(testWordLists, testTagLists, wordDict, tagDict,2)
check(language="Chinese",gold_path="../NER/Chinese/chinese_test.txt",my_path="./output_Chinese.txt")