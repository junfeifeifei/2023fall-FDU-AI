from CRF import *
from check import check

wordDict = acquireDict(['../NER/English/train.txt'])
tagDict = {'O': 0,
           'B-PER': 1, 'I-PER': 2,
           'B-ORG': 3, 'I-ORG': 4,
           'B-LOC': 5, 'I-LOC': 6,
           'B-MISC': 7, 'I-MISC': 8}

trainWordLists, trainTagLists = prepareData('../NER/English/train.txt')
testWordLists, testTagLists = prepareData('../NER/English/english_test.txt')

crf = CRFModel(1)
crf.train(trainWordLists, trainTagLists)
crf.test(testWordLists, testTagLists, wordDict, tagDict, 1)
check(language="English", gold_path="../NER/English/english_test.txt", my_path="./output_English.txt")
