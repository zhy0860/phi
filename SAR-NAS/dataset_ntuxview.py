import os
import scipy.io as sio
from torchvision import transforms,utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
def default_loader(path):
    return sio.loadmat(path)
class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        fh = open(txt, 'r')
        skeleton = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            skeleton.append((words[0], int(words[1])))
        self.skeleton = skeleton
        self.transform = transform
        self.loader = loader
        self.txt = txt
    def __getitem__(self, index):
        fn, lable = self.skeleton[index]
        phase = self.txt.split('/')[-1][:-4]
        skeletons = self.loader(os.path.join('/media/lab540/79eff75a-f78c-42f2-8902-9358e88bf654/lab540/Neura_auto_search/datasets/ntu112/ntu_cv/',\
                                             phase, fn))['final_matrix']
        if self.transform:
            skeletons = self.transform(skeletons)
        skeletons = skeletons.float()
        return skeletons, lable

    def __len__(self):
        return len(self.skeleton)
if __name__ == "__main__":
    data = MyDataset(txt='/media/lab540/79eff75a-f78c-42f2-8902-9358e88bf654/lab540/Neura_auto_search/datasets/ntu112/ntu_cv/train.txt', transform=None)
    valdata = MyDataset(txt='/media/lab540/79eff75a-f78c-42f2-8902-9358e88bf654/lab540/Neura_auto_search/datasets/ntu112/ntu_cv/test.txt', transform=None)
    print(data.__len__(), valdata.__len__())