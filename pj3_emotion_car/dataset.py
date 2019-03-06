import torch
import numpy as np
import pj3_emotion_car.config as conf


class Dataset:  # data for aspect_polarity:
    def __init__(self, training=True):
        self.training = training
        # data
        path = conf.train_data if training else conf.dev_data
        data = np.load(path)

        self.sentences = [self.to_tenser(text) for text in data['texts']]
        self.labels = [self.label2tenser(label) for label in data['polars']]
        self.targets = [self.label2tenser(label) for label in data['aspects']]
        print('data size: {}'.format(len(self.labels)))
        del data

    def to_tenser(self, text):
        tenser = torch.from_numpy(np.asarray(text))
        return tenser.view(tenser.size()[0], -1).long()

    def label2tenser(self, label):
        tenser = torch.from_numpy(np.asarray([label]))
        return tenser.view(-1).long()

    def get(self, index):
        line_tensor = self.sentences[index]
        line_tensor = line_tensor.cuda()
        category_tensor = self.labels[index].cuda()
        elmo_tensor = None
        target_tensor = self.targets[index].cuda()

        return (line_tensor, elmo_tensor, target_tensor), category_tensor
