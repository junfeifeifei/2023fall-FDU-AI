from collections import OrderedDict


def prepareData(filePath):
    f = open(filePath, 'r', encoding='utf-8', errors='ignore')
    wordlists, taglists = [], []
    wordlist, taglist = [], []
    for line in f.readlines():
        if line == '\n':
            wordlists.append(wordlist);
            taglists.append(taglist)
            wordlist, taglist = [], []
        else:
            word, tag = line.strip('\n').split()
            wordlist.append(word);
            taglist.append(tag)
    if len(wordlist) != 0 or len(taglist) != 0:
        wordlists.append(wordlist);
        taglists.append(taglist)
    f.close()
    return wordlists, taglists


'''
数字标识转化为字符串
: origin是原来的数字标识(字典标识)
: dictionary 是对应的字典
'''


def int2str(origin, dictionary):
    result = []
    keys = list(dictionary.keys())
    if isinstance(origin[0], list):
        for i in range(len(origin)):
            result.append([])
            for j in range(len(origin[i])):
                result[i].append(keys[int(origin[i][j])])
    else:
        for i in range(len(origin)):
            result.append(keys[int(origin[i])])
    return result



def str2int(origin, dictionary):
    # print(dictionary)
    result = []
    if isinstance(origin[0], list):
        for i in range(len(origin)):
            result.append([])
            for j in range(len(origin[i])):
                if origin[i][j] in dictionary:
                    result[i].append(dictionary[origin[i][j]])
                else:
                    result[i].append(dictionary['UNK'])
    else:
        for i in range(len(origin)):
            if origin[i] in dictionary:
                result.append(dictionary[origin[i]])
            else:
                result.append(dictionary['UNK'])
    return result



def acquireDict(fileNameList):
    wordDict = OrderedDict()
    wordDict['UNK'] = 0
    for fileName in fileNameList:
        f = open(fileName, 'r', encoding='utf-8', errors='ignore')
        for line in f.readlines():
            if line == '\n': continue
            word, tag = line.strip('\n').split()

            if word not in wordDict:
                wordDict[word] = len(wordDict)
        f.close()
    return wordDict


def match(listOne, listTwo):
    result = []
    for element in listOne:
        if element in listTwo:
            result.append(element)
    return result


def word2features(sent, i):
    word = sent[i]
    prev_word = '<s>' if i == 0 else sent[i - 1]
    next_word = '</s>' if i == (len(sent) - 1) else sent[i + 1]
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word + word,
        'w:w+1': word + next_word,
        'bias': 1
    }
    return features


def sent2features(sent):
    """抽取序列特征"""
    return [word2features(sent, i) for i in range(len(sent))]
