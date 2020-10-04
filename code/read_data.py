import os
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from tqdm import tqdm

from sklearn.neighbors import KernelDensity

import gensim
import torch
import torch.utils.data as Data
import torchtext.vocab as vocab
from torch.utils.data import Dataset
import torch.nn.functional as F

from utils import sentence_tokenize, transform_format, check_ack_word

def read_all_unlabeled(limited_unlabeled_data):

    with open(data_path + 'labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
        # {mid: sentences, labels}
    with open(data_path + 'unlabeled_data.pkl', 'rb') as f:
        unlabeled_data = pickle.load(f)
        # {mid: message}
    with open(data_path + 'mid2target.pkl', 'rb') as f:
        mid2target = pickle.load(f)
        # {mid: target, team_size}
    
    with open(data_path + 'label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)

    print(label_mapping)
    try:
        with open(data_path + 'vocab_2.pkl', 'rb') as f:
            vocab = pickle.load(f)

        print('unk words: ', vocab.unk_count)
        print('vocab size: ', vocab.vocab_size)
    except:
        vocab = Vocab(unlabeled_data=unlabeled_data,
                      labeled_data=labeled_data, embedding_size=embedding_size)

        with open(data_path + 'vocab_2.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        print('unk words: ', vocab.unk_count)
        print('vocab size: ', vocab.vocab_size)

    all_ids = [k for k in limited_unlabeled_data]
    
    train_unlabeled_dataset = Loader_unlabeled(
        vocab, limited_unlabeled_data, all_ids, mid2target, max_seq_num, max_seq_len)
    return train_unlabeled_dataset

def read_data(data_path, n_labeled_data=300, n_unlabeled_data=-1, max_seq_num=8, max_seq_len=64, embedding_size=128):
    with open(data_path + 'labeled_data.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
        # {mid: sentences, labels}
    with open(data_path + 'unlabeled_data.pkl', 'rb') as f:
        unlabeled_data = pickle.load(f)
        # {mid: message}
    with open(data_path + 'mid2target.pkl', 'rb') as f:
        mid2target = pickle.load(f)
        # {mid: target, team_size}
    
    with open(data_path + 'label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)

    print(label_mapping)
    try:
        with open(data_path + 'vocab_2.pkl', 'rb') as f:
            vocab = pickle.load(f)

        print('unk words: ', vocab.unk_count)
        print('vocab size: ', vocab.vocab_size)
    except:
        vocab = Vocab(unlabeled_data=unlabeled_data,
                      labeled_data=labeled_data, embedding_size=embedding_size)

        with open(data_path + 'vocab_2.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        print('unk words: ', vocab.unk_count)
        print('vocab size: ', vocab.vocab_size)

    np.random.seed(1)
    labeled_data_ids = list(labeled_data.keys())
    np.random.shuffle(labeled_data_ids)
    unlabeled_data_ids = list(unlabeled_data.keys())
    np.random.shuffle(unlabeled_data_ids)

    if len(labeled_data_ids) > 1000:
        n_labeled_data = min(len(labeled_data_ids)-800, n_labeled_data)
    else:
        n_labeled_data = min(len(labeled_data_ids)-500, n_labeled_data)
    
    train_labeled_ids = labeled_data_ids[:n_labeled_data]
    if n_unlabeled_data == -1:
        n_unlabeled_data = len(unlabeled_data_ids)
    train_unlabeled_ids = unlabeled_data_ids[:n_unlabeled_data]

    if len(labeled_data_ids) > 1000:
        val_ids = labeled_data_ids[-800:-400]
        test_ids = labeled_data_ids[-400:]
    else:
        val_ids = labeled_data_ids[-500:-300]
        test_ids = labeled_data_ids[-300:]
    
    train_labeled_dataset = Loader_labeled(
        vocab, labeled_data, train_labeled_ids, mid2target, label_mapping, max_seq_num, max_seq_len)
    train_unlabeled_dataset = Loader_unlabeled(
        vocab, unlabeled_data, train_unlabeled_ids, mid2target, max_seq_num, max_seq_len)

    val_dataset = Loader_labeled(
        vocab, labeled_data, val_ids, mid2target, label_mapping, max_seq_num, max_seq_len)
    test_dataset = Loader_labeled(
        vocab, labeled_data, test_ids, mid2target, label_mapping, max_seq_num, max_seq_len)

    n_class_sentence = 0
    for (u,v) in label_mapping.items():
        if v!= 0:
            n_class_sentence += 1
    n_class_sentence += 1

    doc_label = []
    for (u,v) in mid2target.items():
        doc_label.append(v)
    n_class_doc = max(doc_label) + 1

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}, N class {}, {}".format(
        len(train_labeled_ids), len(train_unlabeled_ids), len(val_ids), len(test_ids), n_class_sentence, n_class_doc))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, vocab, n_class_sentence, n_class_doc


class Vocab(object):
    def __init__(self, unlabeled_data=None, labeled_data=None, embedding_size=128, max_seq_num=6, max_seq_len=128):
        self.word2id = {}
        self.id2word = {}

        self.pattern = r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """
        self.english_punctuations = []
        #    ',', '.', ':', ';', '(', ')', '[', ']', '@', '#', '%', '*', '\"', '=', '^', '_', '~', '-']

        self.build_vocab(unlabeled_data, labeled_data,
                         embedding_size, max_seq_num, max_seq_len)

        self.vocab_size = len(self.word2id)

        self.embed = self.build_embed_matrix(embedding_size)
    
    def build_vocab(self, unlabeled_data, labeled_data, embedding_size, max_seq_num, max_seq_len):
        sentences = []
        words = []
        if unlabeled_data is not None:
            for (u, v) in unlabeled_data.items():
                try:
                    results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                    dd = results.sub(" <website> ", v)
                    results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                    dd = results.sub(" <website> ", dd)
                    results = re.compile(r'[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)', re.S)
                    dd = results.sub(" <website> ", dd)
                    sents = sentence_tokenize(dd)
                    for j in range(0, len(sents)):
                        a = regexp_tokenize(
                            transform_format(sents[j]), self.pattern)
                        temp = []
                        for k in range(0, len(a)):
                            if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                                if a[k].isdigit():
                                    a[k] = '<number>'
                                elif a[k][0] == '$':
                                    a[k] = '<money>'
                                elif a[k][-1] == '%':
                                    a[k] = '<percentage>'
                                temp.append(a[k].lower())
                                words.append(a[k].lower())
                        if len(temp) > 0:
                            sentences.append(temp)
                except:
                    #print(u,v)
                    #exit()
                    pass

        if labeled_data is not None:
            for (u, v) in labeled_data.items():
                for i in range(0, len(v[0])):
                    v[0][i] = str(v[0][i])
                    try:
                        results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                        dd = results.sub(" <website> ", v[0][i])
                        results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                        dd = results.sub(" <website> ", dd)
                        results = re.compile(r'[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)', re.S)
                        dd = results.sub(" <website> ", dd)
                    except:
                        print(u, v)
                        print(v[0][i])
                        exit()
                    a = regexp_tokenize(transform_format(dd), self.pattern)
                    temp = []
                    for k in range(0, len(a)):
                        if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                            if a[k].isdigit():
                                a[k] = '<number>'
                            elif a[k][0] == '$':
                                a[k] = '<money>'
                            elif a[k][-1] == '%':
                                a[k] = '<percentage>'
                            temp.append(a[k].lower())
                            words.append(a[k].lower())
                    if len(temp) > 0:
                        sentences.append(temp)

        word_frequency = {}
        for i in range(0, len(words)):
            if words[i] in word_frequency:
                word_frequency[words[i]] += 1
            else:
                word_frequency[words[i]] = 1

        self.model = gensim.models.Word2Vec(
            sentences, size=embedding_size, window=5, min_count=1, iter=20, negative=50)

        x = 4
        self.word2id['<pad>'] = 0
        self.id2word[0] = '<pad>'
        self.word2id['<sos>'] = 2
        self.id2word[2] = '<sos>'
        self.word2id['<eos>'] = 3
        self.id2word[3] = '<eos>'

        self.unk_count = 0

        for i in range(0, len(sentences)):
            for j in range(0, len(sentences[i])):
                if word_frequency[sentences[i][j].lower()] >= 2:
                    if sentences[i][j].lower() in self.model:
                        if sentences[i][j].lower() in self.word2id:
                            pass
                        else:
                            self.word2id[sentences[i][j].lower()] = x
                            self.id2word[x] = sentences[i][j].lower()
                            x = x + 1
                else:
                    self.word2id['<unk>'] = 1
                    self.id2word[1] = '<unk>'
                    self.unk_count += 1

    def build_embed_matrix(self, embedding_size):
        X = np.random.normal(loc=0.0, scale=1.0, size=[
                             self.vocab_size, embedding_size])
        X[0] = np.zeros([1, embedding_size])
        #X[0] = np.zeros([1, embedding_size])
        #X[0] = np.zeros([1, embedding_size])
        for (u, v) in self.id2word.items():
            if v in self.model:
                vector = self.model.wv[v]
                X[u] = vector
        return X

    

class Loader_labeled(Dataset):
    def __init__(self, vocab, labeled_data, ids, target, label_set, max_seq_num=8, max_seq_len=64):
        self.vocab = vocab
        self.labeled_data = labeled_data
        self.ids = ids
        self.target = target
        self.max_seq_len = max_seq_len
        self.max_seq_num = max_seq_num

        self.pattern = r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                       """
        self.english_punctuations = []
        #    ',', '.', ':', ';', '(', ')', '[', ']', '@', '#', '%', '*', '\"', '=', '^', '_', '~', '-']
        
        self.label_set = label_set
        
        self.kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        
        self.load_data(labeled_data, ids)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        mid = self.ids[idx]
        sents, l, sent_len, doc_len= self.message[mid]

        message_target = self.lookup_score(mid)
        labels = np.array([10] * self.max_seq_num)
        doc_len = np.array(doc_len)
        sent_length = np.array([0] * self.max_seq_num)

        # select labeled sent
        mask1 = np.array([0] * self.max_seq_num)
        # select unlabeled sent
        mask2 = np.array([0] * self.max_seq_num)
        # select padded sent
        mask3 = np.array([1] * self.max_seq_num)
        # select unpadded sent
        mask4 = np.array([0] * self.max_seq_num)

        for i in range(0, len(l)):
            labels[i] = l[i]
            sent_length[i] = sent_len[i]
            if l[i] != 10:
                mask1[i] = 1
                mask2[i] = 0
                mask3[i] = 0
                mask4[i] = 1
            if l[i] == 10:
                mask1[i] = 0
                mask2[i] = 1
                mask3[i] = 0
                mask4[i] = 1
        
        message_vec = torch.LongTensor(self.message2id(sents))

        return (message_vec, labels, message_target, mask1, mask2, mask3, mask4, mid, sent_length, doc_len)

    
    def lookup_score(self, id):
        return self.target[id]
    
    def lookup_label_id(self, s):
        return self.label_set[s]
    
    def message2id(self, message):
        X = np.zeros([self.max_seq_num, self.max_seq_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_seq_num and j < self.max_seq_len:
                    try:
                        id = self.vocab.word2id[si.lower()]
                        X[i][j] = id
                    except:
                        X[i][j] = 1
        
        for i in range(len(message), self.max_seq_num):
            X[i][0] = 2
            X[i][1] = 3

        return X
    
    def load_data(self, labeled_data, ids):
        self.message = {}
        
        labels_esit = []
        
        for i in ids:
            sentences = []
            labels = []
            doc_len = []
            sent_len = []

            sents, l = labeled_data[i]

            for j in range(0, len(sents)):
                
                sents[j] = str(sents[j])
                
                results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", sents[j])
                results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", dd)
                results = re.compile(r'[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)', re.S)
                dd = results.sub(" <website> ", dd)

                a = regexp_tokenize(transform_format(dd), self.pattern)

                temp = []
                for k in range(0, len(a)):
                    if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                        if a[k].isdigit():
                            a[k] = '<number>'
                        elif a[k][0] == '$':
                            a[k] = '<money>'
                        elif a[k][-1] == '%':
                            a[k] = '<percentage>'
                        temp.append(a[k].lower())

                if len(temp) > 0:
                    temp_ = ['<sos>']
                    for k in range(0, min(len(temp), self.max_seq_len -2)):
                        temp_.append(temp[k])
                    temp_.append('<eos>')
                    sentences.append(temp_)
                    labels.append(self.lookup_label_id(l[j]))
                    
                    labels_esit.append(self.lookup_label_id(l[j]))
                    
                    sent_len.append(len(temp_) - 1)
            
            doc_len.append(len(sents) - 1)
            
            self.message[i] = (sentences, labels, sent_len, doc_len)  
            
        x_d = set()
        for (u, v) in self.label_set.items():
            x_d.add(v)
        x_d = np.array(list(x_d))
        
        
        self.kde.fit(np.array(labels_esit)[:, None])
        self.dist = self.kde.score_samples(x_d[:, None])

        
        self.esit_dist = F.softmax(torch.tensor(self.dist), dim = -1)
    


class Loader_unlabeled(Dataset):
    def __init__(self, vocab, unlabeled_data, ids, target, max_seq_num=8, max_seq_len=64):
        self.vocab = vocab
        self.unlabeled_data = unlabeled_data
        #self.ids = ids
        self.target = target
        self.max_seq_num = max_seq_num
        self.max_seq_len = max_seq_len

        self.pattern = r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                       """
        self.english_punctuations = []
        #    ',', '.', ':', ';', '(', ')', '[', ']', '@', '#', '%', '*', '\"', '=', '^', '_', '~', '-']
        self.load_data(unlabeled_data, ids)
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mid = self.ids[idx]
        sents, l, sent_len, doc_len  = self.message[mid]

        message_target = self.lookup_score(mid)
        
        doc_len = np.array(doc_len)

        sent_length = np.array([0] * self.max_seq_num)

        labels = np.array([10] * self.max_seq_num)
        # select labeled sent
        mask1 = np.array([0] * self.max_seq_num)
        # select unlabeled sent
        mask2 = np.array([0] * self.max_seq_num)
        # select padded sent
        mask3 = np.array([1] * self.max_seq_num)
        # select unpadded sent
        mask4 = np.array([0] * self.max_seq_num)
        for i in range(0, len(l)):
            labels[i] = l[i]
            sent_length[i] = sent_len[i]
            if l[i] != 10:
                mask1[i] = 1
                mask2[i] = 0
                mask3[i] = 0
                mask4[i] = 1
            if l[i] == 10:
                mask1[i] = 0
                mask2[i] = 1
                mask3[i] = 0
                mask4[i] = 1

        message_vec = torch.LongTensor(self.message2id(sents))

        return (message_vec, labels, message_target, mask1, mask2, mask3, mask4, mid, sent_length, doc_len)



    def message2id(self, message):
        X = np.zeros([self.max_seq_num, self.max_seq_len])
        for i in range(0, len(message)):
            for j, si in enumerate(message[i]):
                if i < self.max_seq_num and j < self.max_seq_len:
                    try:
                        id = self.vocab.word2id[si.lower()]
                        X[i][j] = id
                    except:
                        X[i][j] = 1
        for i in range(len(message), self.max_seq_num):
            X[i][0] = 2
            X[i][1] = 3
        return X
    
    def lookup_score(self, id):
        return self.target[id]
    
    def load_data(self, unlabeled_data, ids):
        self.message = {}
        self.ids = []
        self.data_num = 0

        for i in ids:
            try:
                sentences = []
                labels = []
                doc = unlabeled_data[i]

                doc_len = []
                sent_len = []

                doc += '.'

                results = re.compile(r'http[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", doc)
                results = re.compile(r'www.[a-zA-Z0-9.?/&=:#%_-]*', re.S)
                dd = results.sub(" <website> ", dd)
                results = re.compile(r'[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)', re.S)
                dd = results.sub(" <website> ", dd)

                sents = sentence_tokenize(dd)

                # print(sents)

                for j in range(0, len(sents)):
                    a = regexp_tokenize(
                        transform_format(sents[j]), self.pattern)
                    temp = []
                    for k in range(0, len(a)):
                        if a[k] not in self.english_punctuations and check_ack_word(a[k]) == 1:
                            if a[k].isdigit():
                                a[k] = '<number>'
                            elif a[k][0] == '$':
                                a[k] = '<money>'
                            elif a[k][-1] == '%':
                                a[k] = '<percentage>'
                            temp.append(a[k].lower())

                    if len(temp) > 0:
                        temp_ = ['<sos>']
                        for k in range(0, min(len(temp), self.max_seq_len - 2)):
                            temp_.append(temp[k])
                        temp_.append('<eos>')
                        sentences.append(temp_)
                        labels.append(10)
                        sent_len.append(len(temp_) - 1)

                doc_len.append(min(len(sents) - 1, self.max_seq_num - 1))

                self.message[i] = (sentences[:self.max_seq_num],
                                   labels[:self.max_seq_num], sent_len[:self.max_seq_num], doc_len)
                self.ids.append(i)
                
            except:
                if str(doc) != "nan":
                    print(doc)
                pass
    
