import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from data_preprocess import load_data
from build_network import cnn

def train():
    train_loader, verify_data, test_loader = load_data()
    print('train...')
    epoch_num = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss=1
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss().to(device)
    for epoch in range(epoch_num):
        for batch_idx, img in enumerate(train_loader, 0):
            data, label = img
            running_loss = 0.0
            running_correct = 0
            total=0
            data, label = Variable(data).to(device), Variable(label.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            #print(output)
           # print(label)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            predicted = torch.max(output.data, 1)[1].data
            running_loss += loss.item()
            running_correct += (predicted == label).sum()
            total += label.size(0)
        epoch_loss = running_loss / total
        epoch_acc = running_correct / total
        print('Epoch: {}  Loss: {:.4f} Acc: {:.4f}'.format(epoch,epoch_loss, epoch_acc))
        if (epoch_acc > best_acc) or(epoch_acc == best_acc and best_loss > epoch_loss):
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_model_wts = model.state_dict()


    torch.save(best_model_wts, "../weights/weight_dog_cat.pt")
