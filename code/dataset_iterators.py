from parsed_args import args
import torch
from tqdm import tqdm
from vae_train.vae_utils import *
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# TODO? A lot of these handlers are very similar; however, I think it's simpler to keep them seperate?

def full_iterator(unlabeled_train_iter, unlabeled_trainloader, vocab, model, n_labels):
    content_vectors = []
    strat_vectors = []
    doc_labels = []
    mappings = []
    orig_fetch = []

    while True:
        try:
            try:
                x_u, l_u, y_u, mask1_u, mask2_u, mask3_u, mask4_u, mid_u, sent_len_u, doc_len_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                x_u, l_u, y_u, mask1_u, mask2_u, mask3_u, mask4_u, mid_u, sent_len_u = unlabeled_train_iter.next()
        except:
            break

        x = torch.cat([x_u], dim = 0)
        l = torch.cat([l_u], dim = 0)
        y = torch.cat([y_u.long()], dim = 0)

        mask1 = torch.cat([mask1_u], dim = 0)
        mask2 = torch.cat([mask2_u], dim = 0)
        mask3 = torch.cat([mask3_u], dim = 0)
        mask4 = torch.cat([mask4_u], dim = 0)

        doc_len = torch.cat([doc_len_u], dim = 0)
        sent_len = torch.cat([sent_len_u], dim = 0)

        batch_size = l.shape[0]
        seq_num = x.shape[1]
        seq_len = x.shape[2]

        mid = mid_u
        temp = l.view(-1, 1).long()
        l_one_hot = torch.zeros(batch_size*seq_num, n_labels).cuda()

        for i in range(0, len(temp)):
            if temp[i] != 10:
                l_one_hot[i][temp[i]] = 1

        l_one_hot = l_one_hot.view(batch_size, seq_num, n_labels)

        xs, ys = (x.view(batch_size*seq_num, seq_len), l.view(batch_size*seq_num))
        prob = create_generator_inputs(xs, vocab, train = False)

        x, prob, l_one_hot, y, l = x.cuda(), prob.cuda(), l_one_hot.cuda(), y.cuda(), l.cuda()
        mask1, mask2 = mask1.cuda(), mask2.cuda()

        logits, kld_z, q_y, q_y_softmax, t, strategy_embedding, y_in2, content_vec = model(x, prob, 
            args.tau, mask1, mask2, args.hard, l_one_hot, doc_len = doc_len, sent_len = sent_len)

        max_idxs = y_in2.argmax(axis=1)
        argmaxed = torch.zeros(y_in2.shape)
        argmaxed[torch.arange(y_in2.shape[0]),max_idxs] = 1
        y_in2 = (argmaxed.T.cpu() * y_in2.sum(axis=1).cpu()).T

        last_dim = int((content_vec.shape[0] * content_vec.shape[1]) / (batch_size * seq_num))
        content_vectors.append(content_vec.reshape((batch_size, seq_num, last_dim)).tolist())
        curr_strats = y_in2.reshape(batch_size, seq_num, n_labels).tolist()
        
        strat_vectors.append(curr_strats)

        doc_labels.append(y.tolist())
        orig_fetch.append(mid)  

    return content_vectors, strat_vectors, doc_labels, orig_fetch
        
def get_content_strat_vector_details(content_vectors, strat_vectors, doc_labels,
                                     all_mids, attn_content_lstm, return_rate=False):
    
    attns = {
        "content": [],
        "strategy": [],
        "document": []
    }
    
    acc = []
    labels = []
    strategy_orders = []
    all_corr = []
    all_out = []
    with torch.no_grad():
        for i, batch in enumerate(content_vectors):

            sigmoid_out, content_attn,strategy_attn, s_score = attn_content_lstm(
                torch.tensor(content_vectors[i]).cuda().float(), 
                torch.tensor(strat_vectors[i]).cuda().float())

            sigmoid_out = sigmoid_out > .5
            attns["document"].append(s_score)
            attns["content"].append(content_attn)
            attns["strategy"].append(strategy_attn)

            out = sigmoid_out.squeeze().tolist()
            correct = (np.array(doc_labels[i]) == 1).tolist()
            all_corr += correct
            all_out += out
            strategy_orders.append(torch.tensor(strat_vectors[i]))
            labels.append(correct)
            
    # orig-fetch is the same as all mids for the other dataloaders - we technically already compute it, but
    # i'm passing it in again just so the return signature is the same :')
    if return_rate: return f1_score(all_corr, all_out, average="macro"), attns, labels, strategy_orders, (sum(all_out) / len(all_out))
    return f1_score(all_corr, all_out, average="macro"), attns, labels, strategy_orders, all_mids


def get_dataloader_details(dataloader, vae_model, attn_content_lstm, n_labels, vocab):
    
    attns = {
        "content": [],
        "strategy": [],
        "document": []
    }
    
    labels = []
    strat_orders = []
    
    all_correct = []
    all_out = []
    all_mids = []
    
    with torch.no_grad():
        for batch_idx, (x, l, y, mask1, mask2, mask3, mask4, mid, sent_len, doc_len) in \
            tqdm(enumerate(dataloader), position=0, leave=True):

            # first, we're going to run our data through a VAE to get content and strategy.
            batch_size = l.shape[0]
            seq_num = x.shape[1]
            seq_len = x.shape[2]

            temp = l.view(-1, 1).long()
            l_one_hot = torch.zeros(batch_size * seq_num, n_labels).cuda()

            for i in range(0, len(temp)):
                if temp[i] != 10:
                    l_one_hot[i][temp[i]] = 1

            l_one_hot = l_one_hot.view(batch_size, seq_num, n_labels)

            xs, ys = (x.view(batch_size * seq_num, seq_len), l.view(batch_size * seq_num))
            prob = create_generator_inputs(xs, vocab, train = False)

            x, prob, l_one_hot, y, l = x.cuda(), prob.cuda(), l_one_hot.cuda(), y.cuda(), l.cuda()
            mask1, mask2 = mask1.cuda(), mask2.cuda()

            logits, kld_z, q_y, q_y_softmax, t, strategy_embedding, y_in2, content_vec = vae_model(x, 
                prob, args.tau, mask1, mask2, args.hard, l_one_hot, doc_len = doc_len, sent_len = sent_len)
            
            last_dim = int((content_vec.shape[0] * content_vec.shape[1]) / (batch_size * seq_num))
            content_vec = content_vec.reshape((batch_size, seq_num, last_dim))
            y_in2 = y_in2.reshape(batch_size, seq_num, n_labels)
            
            # next, we're going to pass it through our LSTM
            sigmoid_out, content_attn,strategy_attn, s_score = attn_content_lstm(content_vec, y_in2)

            strat_orders.append(y_in2)
            sigmoid_out = sigmoid_out > .5
            attns["document"].append(s_score)
            attns["content"].append(content_attn)
            attns["strategy"].append(strategy_attn)
            
            # "mid" is just a message id -- this is useful if we want to go pick out samples from our dataset.
            all_mids.append(mid)

            out = sigmoid_out.squeeze().tolist()
            correct = (y == 1).tolist()
            labels.append(correct)
            
            all_correct += correct
            all_out += out
            
        
        print(str(f1_score(all_correct, all_out, average="macro")) + " f1")
        print(str(precision_score(all_correct, all_out, average="macro")) + " p")
        print(str(recall_score(all_correct, all_out, average="macro")) + " r")
        print(str(accuracy_score(all_correct, all_out)) + " acc")
        print(str(roc_auc_score(all_correct, all_out)) + " roc auc")


        return f1_score(all_correct, all_out, average="macro"), attns, labels, strat_orders, all_mids