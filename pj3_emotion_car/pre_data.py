
import pj3_emotion_car.config as conf
import jieba
import numpy as np
import json

from sklearn.model_selection import train_test_split


def get_vocab():
    words = ['舒适']
    with open(conf.raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            splits = line.rstrip().split(',')
            text = splits[1]
            aspect = splits[2]
            texts = jieba.lcut(text)
            words.append(aspect)
            words.extend(texts)
    words = set(words)
    return words


def pre_emb_mat(vocab):
    emb_mat = np.zeros((len(vocab), conf.emb_size), dtype='float32')
    data = np.load(conf.raw_emb_path)
    w2d = data['word2id'].item()
    vec = data['vec']
    word2id = {}
    for idx, word in enumerate(vocab):
        word2id[word] = idx
        if word in w2d:
            emb_mat[idx] = np.asarray(vec[w2d[word]], dtype='float32')
        else:
            emb_mat[idx] = np.asarray(np.random.random(conf.emb_size), dtype='float32')
    word2id['<pad>'] = len(word2id)
    emb_mat = np.vstack((emb_mat, np.asarray([0]*conf.emb_size, dtype='float32')))

    aspect2id = np.load(conf.label2id)['aspect2id'].item()
    aspect_emb = np.zeros((len(aspect2id), conf.emb_size), dtype='float32')
    for aspect, idx in aspect2id.items():
        aspect = aspect if aspect != '舒适性' else '舒适'
        aspect_emb[idx] = emb_mat[word2id[aspect]]

    np.savez_compressed(conf.emb_path, vec=emb_mat, word2id=word2id, aspect_emb=aspect_emb)


def label2id():
    data = json.load(open(conf.attr_path, 'r', encoding='utf-8'))
    aspects = [attr['attribute2'] for attr in data['value']]
    aspect2id = {word: idx for idx, word in enumerate(aspects)}

    data = json.load(open(conf.polar_path, 'r', encoding='utf-8'))
    polars = [attr['attribute2'] for attr in data['value']]
    polar2id = {word: idx for idx, word in enumerate(polars)}

    np.savez_compressed(conf.label2id, aspect2id=aspect2id, polar2id=polar2id)


def data2id():
    aspect2id = np.load(conf.label2id)['aspect2id'].item()
    polars2id = np.load(conf.label2id)['polar2id'].item()

    data = np.load(conf.emb_path)
    vec = data['vec']
    word2id = data['word2id'].item()
    del vec

    texts = []
    aspects = []
    polars = []
    with open(conf.raw_data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            splits = line.rstrip().split(',')
            aspect = splits[2]
            polar = splits[3]
            text = [word2id.get(word) for word in jieba.lcut(splits[1])]
            aspects.append([aspect2id[aspect]])
            polars.append([polars2id[polar]])
            texts.append(np.array(text))

    train_texts, val_texts, train_aspects, val_aspects, train_polars, val_polars = \
        train_test_split(texts, aspects, polars, test_size=0.2, random_state=2019)

    np.savez_compressed(conf.train_data, texts=train_texts, aspects=train_aspects, polars=train_polars)
    np.savez_compressed(conf.dev_data, texts=val_texts, aspects=val_aspects, polars=val_polars)


if __name__ == '__main__':
    import os
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # label2id
    label2id()

    # 获取词汇表 并简化embedding矩阵
    vocab = get_vocab()
    pre_emb_mat(vocab)

    # data2id
    data2id()


