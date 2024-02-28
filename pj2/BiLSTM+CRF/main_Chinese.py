from util import *
import pickle
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from BiLSTM_CRF import BiLSTM_CRF
from check import check

Chinese_TagDict = {
    'O': 0,
    'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 'S-NAME': 4
    , 'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8
    , 'B-EDU': 9, 'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12
    , 'B-TITLE': 13, 'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16
    , 'B-ORG': 17, 'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20
    , 'B-RACE': 21, 'M-RACE': 22, 'E-RACE': 23, 'S-RACE': 24
    , 'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 'S-PRO': 28
    , 'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32
}


def save_model(saved_model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(saved_model, file)


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def train(model, epochs, device, train_dataloader, lr=0.001, weight_decay=1e-4, save_path='./model.pkl'):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        model.state = 'train'
        for step, (text, label, seq_len) in enumerate(train_dataloader, start=1):
            text = text.to(device)
            label = label.to(device)
            seq_len = seq_len.to(device)
            loss = model(text, seq_len, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch: [{epoch + 1}/{epochs}],'
                  f'  cur_epoch_finished: {step / len(train_dataloader) * 100:2.2f}%,'
                  f'  loss: {loss.item():2.4f},')
            save_model(model, save_path)


def train_BiLSTM_CRF_Chinese():
    torch.manual_seed(3280)
    embedding_size = 128
    hidden_dim = 768
    batch_size = 32
    device = "cpu"
    wordDict = acquireDict(['../NER/Chinese/train.txt'])
    tagDict = Chinese_TagDict
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tagDict[START_TAG] = len(tagDict)
    tagDict[STOP_TAG] = len(tagDict)
    train_dataset = Mydataset('../NER/Chinese/train.txt', wordDict, tagDict)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True,
                                  collate_fn=train_dataset.collect_fn)
    model = BiLSTM_CRF(embedding_size, hidden_dim, train_dataset.vocab, train_dataset.label_map, device).to(device)
    train(model, 50, device, train_dataloader, 0.001, 1e-4, './BiLSTM_CRF_Chinese.pkl')


def test_BiLSTM_CRF_Chinese():
    batch_size = 32
    wordDict = acquireDict(['../NER/Chinese/train.txt'])
    tagDict = Chinese_TagDict
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tagDict[START_TAG] = len(tagDict)
    tagDict[STOP_TAG] = len(tagDict)
    valid_dataset = Mydataset('../NER/Chinese/chinese_test.txt', wordDict, tagDict)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, pin_memory=False, shuffle=False,
                                  collate_fn=valid_dataset.collect_fn)
    model = load_model('./BiLSTM_CRF_Chinese.pkl')
    device = 'cpu'
    all_pred = []
    model.eval()
    model.state = 'eval'
    with torch.no_grad():
        for text, label, seq_len in tqdm(valid_dataloader, desc='eval: ', disable=True):
            text = text.to(device)
            seq_len = seq_len.to(device)
            batch_tag = model(text, seq_len, label)
            all_pred.extend(int2str(batch_tag, tagDict))
    f = open('output_Chinese.txt', 'w')
    for sentence_str, tagPre_str in zip(valid_dataset.trainWordLists, all_pred):
        for i in range(len(sentence_str)):
            f.write(sentence_str[i])
            f.write(' ')
            f.write(tagPre_str[i])
            f.write('\n')
        f.write('\n')
    f.close()


# train_BiLSTM_CRF_Chinese()
test_BiLSTM_CRF_Chinese()
check(language="Chinese",gold_path="../NER/Chinese/chinese_test.txt",my_path="./output_Chinese.txt")