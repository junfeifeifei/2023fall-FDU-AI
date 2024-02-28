from collections import OrderedDict
import torch
from torch.utils.data import Dataset


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
                result[i].append(dictionary.get(origin[i][j], 0))
    else:
        for i in range(len(origin)):
            result.append(dictionary.get(origin[i], 0))
    return result


def acquireDict(fileNameList):
    wordDict = OrderedDict()
    wordDict['UNK'] = 0
    wordDict['PAD'] = 1
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


class Mydataset(Dataset):
    def __init__(self, file_path, vocab, label_map):
        self.file_path = file_path
        self.trainWordLists, self.trainTagLists = prepareData(file_path)
        self.vocab = vocab
        self.label_map = label_map
        self.examples = []
        for text, label in zip(self.trainWordLists, self.trainTagLists):
            t = [self.vocab.get(t, 0) for t in text]
            l = [self.label_map[l] for l in label]
            self.examples.append([t, l])

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

    def collect_fn(self, batch):
        text = [t for t, l in batch]
        label = [l for t, l in batch]
        seq_len = [len(i) for i in text]
        max_len = max(seq_len)

        text = [t + [self.vocab['PAD']] * (max_len - len(t)) for t in text]
        label = [l + [self.label_map['O']] * (max_len - len(l)) for l in label]

        text = torch.tensor(text, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        seq_len = torch.tensor(seq_len, dtype=torch.long)

        return text, label, seq_len
