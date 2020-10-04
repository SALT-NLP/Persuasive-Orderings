import torch
import torch.nn as nn

def train_model(model, content_vectors, strat_vectors, doc_labels, learning_rate=.001):
    
    test_model.train()
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(test_model.parameters(), lr = learning_rate)

    losses = []
    val_accs = []

    val_acc = generic_eval(val_loader)
    print(val_acc[0])

    best_f1 = float('-inf')

    for epoch in range(0, 1000):
        test_model.train()
        for i, batch in enumerate(content_vectors):

            strat_torch = torch.tensor(strat_vectors[i]).cuda().float()
            content_torch = torch.tensor(content_vectors[i]).cuda().float()
            optimizer.zero_grad()

            sigmoid_out, _, _, _, = test_model(
                            torch.tensor(content_vectors[i]).cuda().float(), 
                            strat_torch)


            loss = criterion(sigmoid_out, torch.tensor(doc_labels[i]).cuda().float().unsqueeze(1))
            loss.backward()
            optimizer.step()

        print()
        losses.append(loss)
        print(str(epoch) + " epoch")
        val_acc = generic_eval(val_loader)
        print(str(best_f1) + " best f1")
        print(str(val_acc[0]) + " curr f1")
        if val_acc[0] > best_f1:
            best_f1 = val_acc[0]
            print("SAVING BEST BORROW MODEL")
            torch.save(test_model.state_dict(), "best_borrow_extra_model.pkl")
        val_accs.append(val_acc[0])
        print(str(loss) + " loss")
