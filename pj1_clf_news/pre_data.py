import pj1_clf_news.config as conf
import numpy as np
import re
import jieba
import os


def write(datas, path):
    if len(datas) > 0:
        with open(path, 'a', encoding='utf-8') as fw:
            fw.write('\n'.join(datas))
            fw.write('\n')


# 第二步： 处理预训练的embedding
def emb2mat(vocab):
    word2id = {word: idx + 2 for idx, word in enumerate(vocab)}
    word2id["<PAD>"] = 0
    word2id["<UNK>"] = 1
    emb_mat = np.zeros((len(word2id), conf.emb_size))
    vec_dict = dict()
    print('word2id size: {}'.format(len(word2id)))
    with open(conf.raw_emb_path, 'r', encoding='utf-8') as f:
        for line in f:
            lines = line.rstrip().split()
            if len(lines) == conf.emb_size + 1:
                word = lines[0]
                if word in word2id:
                    vec_dict[word] = np.asarray(lines[1:], dtype='float32')
    print('vec_dict size: {}'.format(len(vec_dict)))
    # 根据字典生成词向量矩阵
    for word, id in word2id.items():
        vector = vec_dict.get(word, np.asarray(np.random.random(conf.emb_size), dtype='float32'))
        emb_mat[id] = vector
    np.savez_compressed(conf.emb_path, vec=emb_mat, word2id=word2id, emb_mat=emb_mat)


def cut_padding(text, max_len, vocab):
    # 1保留位置的处理
    # texts = ' '.join([word.rstrip() for word in jieba.lcut(text) if word.rstrip()])
    # text = ' '.join(re.findall("([\u4e00-\u9fff, \t]+)", text)).split()
    # 2多个空格合并 不保留原有位置的处理
    text = ' '.join(re.findall("([\u4e00-\u9fff]+)", text))
    texts = [word.rstrip() for word in jieba.lcut(text) if word.rstrip()]
    for word in texts:
        vocab.add(word)
    texts = texts + ['<PAD>'] * (max_len - len(texts)) if len(texts) < max_len else texts[:max_len]
    texts = ' '.join(texts)
    return texts


# 第一步 清洗数据并分词
def clear_data(src_path=conf.raw_data_path, dst_path=conf.data_file):
    datas = []
    vocab = set()
    with open(src_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            if len(line.split('\t')) != 3:
                print(line)
                continue

            label, title, content = line.split('\t')
            info = '{}\t{}\t{}'.format(label, cut_padding(title, conf.title_seq_len, vocab),
                                       cut_padding(content, conf.content_seq_len, vocab))
            datas.append(info)
            if idx % 1000 == 0:
                print(idx)
                write(datas, dst_path)
                datas = []
        write(datas, dst_path)
    return vocab


def rand_file_data(src_path, dst_path, batch_size):
    """ 大文本数据分批随机打散 """
    # count all lines
    label2id = {}
    with open(src_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            label = line.rstrip().split('\t')[0]
            label2id[label] = label2id.get(label, len(label2id))
    np.savez_compressed(conf.id_vocab_path, label2id=label2id)
    all_num = i + 1
    # shuffle index
    indexs = np.random.permutation(all_num)

    num = (all_num - 1) // batch_size + 1
    for i in range(num):
        # per batch index
        end = min((i + 1) * batch_size, all_num)
        batch_indexs = indexs[i * batch_size: end]

        # shuffle data by batch_index
        index_dict = {index: idx for idx, index in enumerate(batch_indexs)}
        datas, idxs = [], []
        with open(src_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i in index_dict:
                    datas.append(line.rstrip())
                    idxs.append(index_dict[i])

        datas, idxs = np.array(datas), np.array(idxs)
        datas = datas[idxs].tolist()

        write(datas, dst_path)


def data2file(val_ratio=0.2):
    # 将数据分批打散
    temp_path = 'data/temp.txt'
    rand_file_data(conf.data_file, temp_path, 10000)
    # 分成训练集和验证集
    trains, vals = [], []
    with open(temp_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.rstrip()
            if np.random.random() >= val_ratio:
                trains.append(line)
            else:
                vals.append(line)
            if idx % 10000 == 0:
                print(idx)
                write(trains, conf.train_data)
                write(vals, conf.dev_data)
                trains, vals = [], []
        write(trains, conf.train_data)
        write(vals, conf.dev_data)
    os.remove(temp_path)


if __name__ == '__main__':
    import os
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # 1、将文本分词 清洗 padding
    vocab = clear_data()

    # 2、将embedding转换为矩阵, 并简化
    emb2mat(vocab)

    # 3、划分数据集
    data2file(0.3)
