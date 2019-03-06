from torch.utils.data.dataset import Dataset
import numpy as np
import torch as t


class ZhiHuData(Dataset):

    def __init__(self, path):
        self.path = path
        data = np.load(path)
        self.labels = data['label']
        self.titles = data['title']
        self.contents = data['content']
        print('data size: {}'.format(self.titles.shape))
        del data

    def __getitem__(self, index):
        label = t.from_numpy(self.labels[index]).long()
        title = t.from_numpy(self.titles[index]).long()
        content = t.from_numpy(self.contents[index]).long()
        return title, content, label

    def __len__(self):
        return self.labels.shape[0]
