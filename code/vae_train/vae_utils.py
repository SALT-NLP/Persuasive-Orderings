import torch
from parsed_args import args
import numpy as np

if args.warm_up == 'False':
    args.warm_up = False
else:
    args.warm_up = True
    
    
if args.hard == 'False':
    args.hard = False
else:
    args.hard = True
    
def create_generator_inputs(x, vocab, train = True):
    prob = x[:, 0:x.shape[1]-1].clone()
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

        logits, kld_z, q_y, q_y_softmax, t, strategy_embedding, y_in2, content_vec = model(x, prob, args.tau, mask1, mask2, args.hard, l_one_hot, doc_len = doc_len, sent_len = sent_len)
        
        logits, kld_z, q_y, q_y_softmax, t, strategy_embedding = nn.parallel.data_parallel(model, (x, prob, tau, mask1, mask2, args.hard, l_one_hot, doc_len, sent_len))

        mse_loss, likelihood, kld_z, log_prior, classification_loss, kld_y, kld_weight_y, kld_weight_z = criterion(logits, kld_z, q_y, q_y_softmax, t, mask1, mask2, mask3, mask4, x, l, y, l_one_hot, epoch + batch_idx/args.val_iteration, n_labels, dist, tsa_type = args.tsa_type)
        

        kld_weight_y = 0 if kld_y < args.kld_y_thres else kld_weight_y
        class_weight = args.class_weight
        
        
        if args.warm_up:
            predict_weight = linear_rampup(epoch+batch_idx/args.val_iteration) * args.predict_weight
            rec_coef = linear_rampup(epoch+batch_idx/args.val_iteration) * args.rec_coef 

        else:
            predict_weight = args.predict_weight
            rec_coef = args.rec_coef

        
        
        loss = predict_weight * mse_loss + rec_coef * likelihood + class_weight * classification_loss + kld_weight_y * (kld_y + log_prior) + kld_weight_z * kld_z
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
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
                temp = v / predict_dict[u]

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
        
        log_qy = torch.log(q_y_softmax[trainable_idx].float() + 1e-8)
        qy_softmax_selected = q_y_softmax[trainable_idx].float()
        qyg_diff = (log_qy - g.float())
        kld_y = torch.sum((qyg_diff * qy_softmax_selected), dim = -1).mean()

        return mse_loss, likelihood, kld_z, log_prior, classification_loss, kld_y, args.kld_weight_y * linear_rampup(epoch), args.kld_weight_z* linear_rampup(epoch)
