from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from img_read import find_label
from img_read import read_imagepath_from_txt


def Myloader(path):
    return Image.open(path).convert('RGB')


# get a list of paths and labels.
def init_process(path, lens):
    data = []
    #name = find_label(path)
    for i in range(lens[0], lens[1]):
        name = find_label(path[i])
        data.append([path[i], name])

    return data


class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        #print(img.shape)
        return img, label

    def __len__(self):
        return len(self.data)


def load_data():
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    path0='../input/img_paths.txt'
    path=read_imagepath_from_txt(path0)
    train_data = init_process(path, [0, 240]) + init_process(path, [300, 540])
    verify_data = init_process(path, [240, 270]) + init_process(path, [540, 570])
    test_data = init_process(path, [270, 300]) + init_process(path, [570, 600])
    train = MyDataset(train_data, transform=transform, loder=Myloader)
    verify = MyDataset(verify_data, transform=transform, loder=Myloader)
    test = MyDataset(test_data, transform=transform, loder=Myloader)
    train_data = DataLoader(dataset=train, batch_size=10, shuffle=True, num_workers=0)
    verify_data = DataLoader(dataset=verify, batch_size=1, shuffle=True, num_workers=0)
    test_data = DataLoader(dataset=test, batch_size=1, shuffle=True, num_workers=0)
    return train_data, verify_data ,test_data
