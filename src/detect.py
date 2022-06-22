import os
import csv
import torch
from data_preprocess import load_data
from build_network import cnn
from train import train
import time

def test():
    train_loader, verify_load, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load("../weights/weight_dog_cat.pt"), False)
    model.eval()
    total = 0
    current = 0
    f = open('../output/output.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    for data in test_loader:
        time_start = time.time()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
        #print(labels, predicted)
        time_elapsed = time.time()-time_start
        csv_writer.writerow([predicted, time_elapsed])
    f.close()
    print('Accuracy:%d%%' % (100 * current / total))


train()
test()
