
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
import itertools
import numpy as np
from pj4_ner_word.evaluator import Evaluator
from pj4_ner_word.model import ner_model
import pj4_ner_word.config as conf
import pickle
# 反序列化一个自定义的类，需要导入该类 否则会报错AttributeError: Can't get attribute 'CRFDataset'
from pj4_ner_word.pre_data import CRFDataset


seed = 5703958
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

with open(conf.data_path, 'rb') as f:
    datas = pickle.load(f)
    dataset_loader = datas['dataset_loader']
    dev_dataset_loader = datas['dev_dataset_loader']
    CRF_l_map = datas['CRF_l_map']
    f_map = datas['f_map']
    c_map = datas['c_map']
    embedding_tensor = datas['embedding_tensor']
    del datas


def train():
    model = ner_model(len(f_map), len(c_map), CRF_l_map['<start>'], CRF_l_map['<pad>'], len(CRF_l_map), embedding_tensor)
    # optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum)
    optimizer = optim.Adam(model.parameters())
    model.cuda()
    evaluator = Evaluator(CRF_l_map)

    tot_length = sum(map(lambda t: len(t), dataset_loader))
    early_stop_epochs = 0
    best_loss = 0

    for epoch in range(conf.epochs):
        print('# epoch: {}'.format(epoch))
        epoch_loss = 0
        for w_f, tg_v, mask_v, len_v, cnn_features in tqdm(
                itertools.chain.from_iterable(dataset_loader), mininterval=2,
                desc=' - Tot iter %d (epoch %d)' % (tot_length, epoch), leave=False, file=sys.stderr):
            optimizer.zero_grad()
            mlen = len_v.max(0)[0].squeeze()
            w_f = Variable(w_f[:, 0:mlen[1]].transpose(0, 1)).cuda()
            tg_v = Variable(tg_v[:, 0:mlen[1]].transpose(0, 1)).unsqueeze(2).cuda()
            mask_v = Variable(mask_v[:, 0:mlen[1]].transpose(0, 1)).cuda()
            cnn_features = Variable(cnn_features[:, 0:mlen[1], 0:mlen[2]].transpose(0, 1)).cuda().contiguous()

            loss = model(w_f, cnn_features, tg_v, mask_v)
            epoch_loss += loss.view(-1).data.tolist()[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), conf.clip_grad)
            optimizer.step()

        epoch_loss /= tot_length
        print('epoch_loss: ', epoch_loss)

        # validation
        dev_f1_crf, dev_pre_crf, dev_rec_crf, dev_acc_crf = evaluator.calc_score(model, dev_dataset_loader)
        print('dev_f1_crf: %.4f - dev_pre_crf: %.4f - dev_rec_crf: %.4f - dev_acc_crf: %.4f - '
              % (dev_f1_crf, dev_pre_crf, dev_rec_crf, dev_acc_crf))

        # early stop
        if dev_f1_crf > best_loss:
            best_loss = dev_f1_crf
            early_stop_epochs = 0
            torch.save(model, conf.model_path)
        else:
            early_stop_epochs += 1
        if early_stop_epochs > conf.early_stop:
            break


if __name__ == '__main__':
    train()