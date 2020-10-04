import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import Dataset
import math

from model import HierachyVAE
from read_data import *
from utils import *

parser = argparse.ArgumentParser(description='Hierachy VAE')

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default = 6)
parser.add_argument('--batch-size-u', type=int, default=32)
parser.add_argument('--val-iteration', type=int, default=120)


parser.add_argument('--n-highway-layers', type=int, default=0)
parser.add_argument('--encoder-layers', type=int, default=1)
parser.add_argument('--generator-layers', type=int, default=1)
parser.add_argument('--bidirectional', type=bool, default=False)


parser.add_argument('--embedding-size', type=int, default=128)
parser.add_argument('--encoder-hidden-size', type=int, default=128)
parser.add_argument('--generator-hidden-size', type=int, default=128)
parser.add_argument('--z-size', type=int, default=64)

parser.add_argument('--gpu', default='2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled-data', type=int, default=100,
                    help='Number of labeled data')
parser.add_argument('--n-unlabeled-data', type=int, default=-
                    1, help='Number of unlabeled data')

parser.add_argument('--data-path', type=str,
                    default='./borrow/', help='path to data folders')
parser.add_argument('--max-seq-num', type=int, default=6,
                    help='max sentence num in a message')
parser.add_argument('--max-seq-len', type=int, default=64,
                    help='max sentence length')

parser.add_argument('--word-dropout', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--rec-coef', type=float, default=1)
parser.add_argument('--predict-weight', type=float, default=1)
parser.add_argument('--class-weight', type=float, default=5)
parser.add_argument('--kld-weight-y', type=float, default=1)
parser.add_argument('--kld-weight-z', type=float, default=1)

parser.add_argument('--kld-y-thres', type=float, default=1.4)
parser.add_argument('--warm-up', default='False', type=str)
parser.add_argument('--hard', type=str, default='False')
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--tau-min', type=float, default=0.4)
parser.add_argument('--anneal-rate', type=float, default=0.01)

parser.add_argument('--tsa-type', type=str, default='exp')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)


if args.warm_up == 'False':
    args.warm_up = False
else:
    args.warm_up = True
    
    
if args.hard == 'False':
    args.hard = False
else:
    args.hard = True
    


best_acc = 0
total_steps = 0

def main():
    global best_acc
    

    train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, vocab, n_labels, doc_labels = read_data(
        data_path=args.data_path, n_labeled_data=args.n_labeled_data, n_unlabeled_data=args.n_unlabeled_data, max_seq_num=args.max_seq_num, max_seq_len=args.max_seq_len, embedding_size=args.embedding_size)
    
    dist = train_labeled_dataset.esit_dist

    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_dataset, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_dataset, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_dataset, batch_size=16, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_dataset, batch_size=16, shuffle=False)

    model = HierachyVAE(vocab.vocab_size, args.embedding_size, args.n_highway_layers, args.encoder_hidden_size, args.encoder_layers, args.generator_hidden_size, args.generator_layers, args.z_size, n_labels, doc_labels, args.bidirectional, vocab.embed, args.hard).cuda()
    model = nn.DataParallel(model)

    train_criterion = HierachyVAELoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(params = filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)
    
    test_accs = []
    
    count = 20

    for epoch in range(args.epochs):
        
        if epoch % 10 == 0:
            args.tau = np.maximum(args.tau * np.exp(-args.anneal_rate*epoch), args.tau_min)
        
        train(labeled_trainloader, unlabeled_trainloader, vocab, optimizer, model, train_criterion, epoch, n_labels, dist)
        _, train_acc, total, macro_f1 = validate(labeled_trainloader, model, criterion, epoch, n_labels, vocab)
        print("epoch {}, train acc {}, train amount {}, micro_f1 {}".format(
            epoch, train_acc, total, macro_f1))
        val_loss, val_acc, total, macro_f1 = validate(val_loader, model, criterion, epoch, n_labels, vocab)
        print("epoch {}, val acc {}, val_loss {}, micro_f1 {}".format(
            epoch, val_acc, val_loss, macro_f1))
        
        count = count -1
        
        if val_acc >= best_acc:
            count = 20
            best_acc = val_acc
            test_loss, test_acc, total, macro_f1 = validate(test_loader, model, criterion, epoch, n_labels, vocab)
            test_accs.append((test_acc, macro_f1))
            
            torch.save(model, args.data_path + 'model.pkl')
            
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))
            
        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)
        
        if count < 0:
            print("early stop")
            break


    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


def create_generator_inputs(x, vocab, train = True):

    prob = []
    for i in range(0, x.shape[0]):
        temp = []
        for j in range(0, x.shape[1]):
            if x[i][j] != 3:
                temp.append(x[i][j])
        prob.append(temp)
    
    prob = torch.tensor(prob)
    
    if train == False:
        return prob

    r = np.random.rand(prob.shape[0], prob.shape[1])
    for i in range(0, prob.shape[0]):
        for j in range(1, prob.shape[1]):
            if r[i, j] < args.word_dropout and prob[i, j] not in [vocab.word2id['<pad>'], vocab.word2id['<eos>']]:
                prob[i, j] = vocab.word2id['<unk>']
    return prob

def train(labeled_trainloader, unlabeled_trainloader, vocab, optimizer, model, criterion, epoch, n_labels, dist):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    tau = args.tau

    for batch_idx in range(args.val_iteration):
        try:
            x, l, y, mask1, mask2, mask3, mask4, mid, sent_len, doc_len = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            x, l, y, mask1, mask2, mask3, mask4, mid, sent_len, doc_len = labeled_train_iter.next()

        try:
            x_u, l_u, y_u, mask1_u, mask2_u, mask3_u, mask4_u, mid_u, sent_len_u, doc_len_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            x_u, l_u, y_u, mask1_u, mask2_u, mask3_u, mask4_u, mid_u, sent_len_u, doc_len_u = unlabeled_train_iter.next()
        
        x = torch.cat([x, x_u], dim = 0)
        l = torch.cat([l, l_u], dim = 0)
        y = torch.cat([y.long(), y_u.long()], dim = 0)

        mask1 = torch.cat([mask1, mask1_u], dim = 0)
        mask2 = torch.cat([mask2, mask2_u], dim = 0)
        mask3 = torch.cat([mask3, mask3_u], dim = 0)
        mask4 = torch.cat([mask4, mask4_u], dim = 0)

        doc_len = torch.cat([doc_len, doc_len_u], dim = 0)
        sent_len = torch.cat([sent_len, sent_len_u], dim = 0)

        batch_size = l.shape[0]
        seq_num = x.shape[1]
        seq_len = x.shape[2]

        temp = l.view(-1, 1).long()
        l_one_hot = torch.zeros(batch_size*seq_num, n_labels).cuda()
        
        for i in range(0, len(temp)):
            if temp[i] != 10:
                l_one_hot[i][temp[i]] = 1
                
        l_one_hot = l_one_hot.view(batch_size, seq_num, n_labels)

        if batch_idx % 30 == 1:
            tau = np.maximum(tau * np.exp(-args.anneal_rate*batch_idx), args.tau_min)
        
        xs, ys = (x.view(batch_size*seq_num, seq_len), l.view(batch_size*seq_num))
        prob = create_generator_inputs(xs, vocab, train = True)

        x, prob, l_one_hot, y, l = x.cuda(), prob.cuda(), l_one_hot.cuda(), y.cuda(), l.cuda()
        mask1, mask2 = mask1.cuda(), mask2.cuda()

        logits, kld_z, q_y, q_y_softmax, t, strategy_embedding = model(x, prob, args.tau, mask1, mask2, args.hard, l_one_hot, doc_len = doc_len, sent_len = sent_len)
        
        mse_loss, likelihood, kld_z, log_prior, classification_loss, kld_y, kld_weight_y, kld_weight_z = criterion(logits, kld_z, q_y, q_y_softmax, t, mask1, mask2, mask3, mask4, x, l, y, l_one_hot, epoch + batch_idx/args.val_iteration, n_labels, dist, tsa_type = args.tsa_type)
        

        if kld_y < args.kld_y_thres:
            kld_weight_y = 0
        else:
            kld_weight_y = kld_weight_y

        
        if classification_loss < 0.001:
            class_weight = args.class_weight
            
        else:
            class_weight = args.class_weight
        
        
        if args.warm_up:
            predict_weight = linear_rampup(epoch+batch_idx/args.val_iteration) * args.predict_weight
        else:
            predict_weight = args.predict_weight
            
        if args.warm_up:
            rec_coef = linear_rampup(epoch+batch_idx/args.val_iteration) * args.rec_coef 
        else:
            rec_coef = args.rec_coef

        
        
        loss = predict_weight * mse_loss + rec_coef * likelihood + class_weight * classification_loss + kld_weight_y * (kld_y + log_prior) + kld_weight_z * kld_z
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx%100 == 0:
            print("epoch {}, step {}, loss {}, mse loss {}, reconstruct {}, classification {}, kld y {}. kld z {}".format(epoch, batch_idx, loss, mse_loss, likelihood, classification_loss, kld_y, kld_z))

def validate(val_loader, model, criterion, epoch, n_labels, vocab):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0
        

        predict_dict = {}
        correct_dict = {}
        correct_total = {}

        for i in range(0, n_labels):
            predict_dict[i] = 0
            correct_dict[i] = 0
            correct_total[i] = 0
        
        p = 0
        r = 0
        for batch_idx, (x, l, y, mask1, mask2, mask3, mask4, mid, sent_len, doc_len) in enumerate(val_loader):
            
            x, l = x.cuda(), l.cuda()

            batch_size = x.shape[0]
            seq_num = x.shape[1]
            seq_len = x.shape[2]
        
            x = x.view(batch_size * seq_num, seq_len)
            l = l.view(batch_size * seq_num).long()
            
            sent_len = sent_len.view(batch_size * seq_num)
            
            logits, ___ = model.module.encode(x, sent_len = sent_len)
            
            _, predicted = torch.max(logits.data, 1)

            trainable_idx = torch.where(mask1.view(batch_size * seq_num) == 1)
            
            
            if len(trainable_idx[0]) <= 0:
                print("...")
                print(mask1.view(batch_size * seq_num))
                print(np.array(mask1.view(batch_size * seq_num)).sum())
                continue
            
            loss = criterion(logits[trainable_idx], l[trainable_idx])
            correct += (np.array(predicted.cpu())[trainable_idx] == np.array(l.cpu())[trainable_idx]).sum()
            input_size = np.array(mask1.view(batch_size * seq_num)).sum()
            loss_total += loss.item() * input_size
            total_sample += input_size

            #print(x.shape, mask1.shape)

            
            for i in range(0, len(trainable_idx[0])):

                correct_total[np.array(l[trainable_idx].cpu())[i]] += 1
                predict_dict[np.array(predicted[trainable_idx].cpu())[i]] += 1

                if np.array(l[trainable_idx].cpu())[i] == np.array(predicted[trainable_idx].cpu())[i]:
                    correct_dict[np.array(l[trainable_idx].cpu())[i]] += 1

        f1 = []
        
        true_total_ = 0
        predict_total_ = 0
        correct_total_ = 0
        
        for (u, v) in correct_dict.items():
            if predict_dict[u] == 0:
                temp = 0
            else:
                temp = v/predict_dict[u]

            if correct_total[u] == 0:
                temp2 = 0
            else:
                temp2 = v/correct_total[u]
            
            if temp == 0 and temp2 == 0:
                f1.append(0)
            else:
                f1.append((2*temp*temp2)/(temp+temp2))
            
            true_total_ += correct_total[u]
            predict_total_ += predict_dict[u]
            correct_total_ += v
        
        Marco_f1 = sum(f1)/(len(f1))
                            
        p =  correct_total_ / predict_total_
        r = correct_total_/ true_total_
                            
        Micro_f1 = (2*p*r)/(p+r)
            
        print('true dist: ', correct_total)
        print('predict dist: ', predict_dict)
        print('correct pred: ', correct_dict)
        print('Macro: ', Marco_f1)
        print('Micro: ', Micro_f1)
                
            
        
        acc_total = correct / total_sample
        loss_total = loss_total / total_sample

    return loss_total, Marco_f1, total_sample, Micro_f1


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def TSA(epoch, n_class, tsa_type = 'exp'):
    epoch = math.floor(epoch)
    if tsa_type == 'exp':
        return np.exp((epoch/args.epochs - 1) * 5) * (1-1/n_class) + 1/n_class
    elif tsa_type == 'linear':
        return epoch/args.epochs * (1- 1/n_class) + 1/n_class
    elif tsa_type == 'log':
        return (1-np.exp(-epoch/args.epochs * 5)) * (1-1/n_class) + 1/n_class
    else:
        return 1


class HierachyVAELoss(object):
    def __call__(self, logits, kld_z, q_y, q_y_softmax, t, mask1, mask2, mask3, mask4, x, l, y, l_one_hot, epoch, n_labels, dist, tsa_type = 'exp'):
        
        mse_loss = F.cross_entropy(t, y.long())

        batch_size = x.shape[0]
        seq_num = x.shape[1]
        seq_len = x.shape[2]
        n_class = l_one_hot.shape[-1]

        xs, ys, ys_one_hot = (x.view(batch_size*seq_num, seq_len), l.view(batch_size*seq_num), l_one_hot.view(batch_size*seq_num, n_class))
        
        xs = xs[:, 1:xs.shape[1]]
        trainable_idx = torch.where(mask4.view(batch_size*seq_num) == 1)

        logits_ = logits[trainable_idx].view(-1, logits.shape[-1])
        xs_ = xs[trainable_idx].contiguous().view(-1)
        weight = torch.tensor([0.0] + [1.0]*(logits.shape[-1]-1)).cuda()
        likelihood = F.cross_entropy(logits_, xs_, weight = weight)

        kld_z = kld_z.mean()

        trainable_idx = torch.where(mask1.view(batch_size * seq_num) == 1)
        prior = standard_categorical(ys_one_hot)
        log_prior = -torch.sum(ys_one_hot[trainable_idx] * torch.log(prior[trainable_idx] + 1e-8), dim = 1).mean()

        thres = TSA(epoch, n_labels, tsa_type)
        
        q_y_log_softmax = F.log_softmax(q_y, dim = 1)

        if len(trainable_idx[0]) > 0:
            count = 0
            classification_loss = 0
            for i in range(0,len(trainable_idx[0])):
                try:
                    if q_y_softmax[trainable_idx[0][i]][ys[trainable_idx[0][i]].long()] < thres:
                        classification_loss += (-1 * q_y_log_softmax[trainable_idx[0][i]][ys[trainable_idx[0][i]].long()])
                        count += 1
                except:
                    print(thres)
                    print(epoch)
                    print(q_y_softmax[trainable_idx[0][i]])
                    print(q_y_softmax[trainable_idx[0][i]][ys[trainable_idx[0][i]].long()])
                    exit()
            if count > 0:
                classification_loss = classification_loss / count
            else:    
                classification_loss = 0
        else:
            classification_loss = 0



        trainable_idx = torch.where(mask2.view(batch_size*seq_num) == 1)
        g = Variable(torch.log(dist)).cuda()
        
        log_qy = torch.log(q_y_softmax[trainable_idx] + 1e-8)
        kld_y = torch.sum(q_y_softmax[trainable_idx]*(log_qy - g), dim = -1).mean()

        return mse_loss, likelihood, kld_z, log_prior, classification_loss, kld_y, args.kld_weight_y * linear_rampup(epoch), args.kld_weight_z* linear_rampup(epoch)
        
if __name__ == '__main__':
    main()

