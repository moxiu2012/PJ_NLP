import numpy as np
from pj2_clfs_zhihu.dataset import ZhiHuData
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd.variable import Variable
from pj2_clfs_zhihu.model import *
import csv

name_model = {'cnn': Cnn, 'cnn_res': CnnRes, 'text_cnn': TextCNN, 'lstm': LSTM, 'rcnn': RCNN,
              'fasttext': FastText}
model_name = 'fasttext'
mode = 'dict'


def batch_test(title, content):
    with t.no_grad():
        title = Variable(title.cuda())
        content = Variable(content.cuda())
        logits = model(title, content)
        probs = t.sigmoid(logits)
    return probs.data.cpu().numpy()


def get_index(preds, remove_zero):
    indexs = []
    for pred in preds:
        index = np.sort(np.argsort(-pred)[: 3])
        if remove_zero:
            index = index[pred[index] > 0]
        indexs.append(index.tolist())
    return indexs


if mode == 'dict':
    data = np.load(conf.emb_dict_path)
    emb_mat = t.from_numpy(data['vec'])
    word2id = data['word2id'].item()
    del data
    vocab_size = len(word2id)

    model_path = conf.model_dict_path.format(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError('model is not exist')

    Model = name_model[model_name]
    model = Model(vocab_size, emb_mat)
    model.load_state_dict(t.load(model_path))
else:
    model = t.load(conf.model_all_path.format(model_name))

model = model.cuda().eval()

dataset = ZhiHuData(conf.test_data)
data_loader = DataLoader(dataset, conf.batch_size)





labels = []
preds = []
for i, batch in tqdm(enumerate(data_loader)):
    title, content, label = batch

    labels.extend(get_index(label.numpy(), True))
    pred = batch_test(title, content)
    preds.extend(get_index(pred, False))

rows = []
for idx, label in enumerate(labels):
    res = ['pred:']
    res.extend(preds[idx])
    res.append('label:')
    res.extend(label)
    rows.append(res)


with open('data/res.csv', 'w') as fw:
    wt = csv.writer(fw)
    wt.writerows(rows)







