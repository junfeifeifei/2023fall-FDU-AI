from collections import OrderedDict


def word2feature(sent, i):
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


def prepareData(filePath):
    f = open(filePath, 'r', encoding='utf-8', errors='ignore')
    wordlists, taglists = [], []
    wordlist, taglist = [], []
    for line in f.readlines():
        if line == '\n':
            wordlists.append(wordlist)
            taglists.append(taglist)
            wordlist, taglist = [], []
        else:
            word, tag = line.strip('\n').split()
            wordlist.append(word)
            taglist.append(tag)
    if len(wordlist) != 0 or len(taglist) != 0:
        wordlists.append(wordlist)
        taglists.append(taglist)
    f.close()
    return wordlists, taglists


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
                result[i].append(dictionary[origin[i][j]])
    else:
        for i in range(len(origin)):
            result.append(dictionary[origin[i]])
    return result


def acquireDict(fileNameList):
    wordDict = OrderedDict()
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


def word2features(sent, i, type):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word': word,
        'word.isdigit()': word.isdigit(),
    }
    template_range = [-3, -2, -1, 0, 1, 2, 3]

    if type == 1:
        for offset in template_range:
            if 0 <= i + offset < len(sent):
                features.update({
                    f'total_word_U{offset:02d}:{sent[i + offset]}': 1.0,
                    f'word.isupper()_U{offset:02d}': sent[i + offset][0].isupper(),
                })
        for offset in template_range:
            if i + offset >= 0 and i + offset + 1 < len(sent):
                features.update({
                    f'B{offset:02d}:{sent[i + offset]}_{sent[i + offset + 1]}': 1.0,
                })
    else:
        for offset in template_range:
            if i + offset >= 0 and i + offset + 1 < len(sent):
                features.update({f'B{offset:02d}:{sent[i + offset][0]}_{sent[i + offset + 1][0]}': 1.0, })
        for offset in template_range:
            if 0 <= i + offset < len(sent):
                features.update({f'U{offset:02d}:{sent[i + offset][0]}': 1.0, })
    return features


def sent2features(sent, type):
    return [word2features(sent, i, type) for i in range(len(sent))]
