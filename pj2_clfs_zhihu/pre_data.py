
import pj2_clfs_zhihu.config as conf
import numpy as np
import word2vec


def emb2npz(emb_file_path, emb_dict_path):
    """将txt格式的embedding转为字典格式, 并将<PAD>和<UNK>加入"""
    emb = word2vec.load(emb_file_path)
    vec = emb.vectors
    word2id = emb.vocab_hash
    word2id['<PAD>'] = len(word2id)
    pad_row = [0] * vec.shape[1]
    vec = np.row_stack((vec, pad_row))
    np.savez_compressed(emb_dict_path, vec=vec, word2id=word2id)
    print('word size: {}'.format(len(word2id)))
    print('emb shape: {}'.format(vec.shape))


def padding(texts, max_len, pad=0):
    texts = texts[:max_len] if len(texts) > max_len else texts + [pad] * (max_len - len(texts))
    return texts


def data2npz(src_path, dst_path):
    """src_path txt: label+\t+title+\t+content
    如：40,6  w6061,w26959,w109   w23255,w728,w12768,w58588,w11,w1442,w855,w36791"""

    data = np.load(conf.emb_path)
    word2id = data['word2id'].item()
    del data

    labels = []
    titles = []
    contents = []
    with open(src_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            label, title, content = line.replace('\n', '').split('\t')

            label = [int(lab) for lab in label.split(',')]
            label_mat = np.zeros(conf.n_classes, dtype='int32')
            label_mat[label] = 1
            labels.append(label_mat)

            # word2id
            title = [word2id[word if word in word2id else '</s>'] for word in title.split(',') if word.rstrip()]
            content = [word2id[word if word in word2id else '</s>'] for word in content.split(',') if word.rstrip()]
            # padding
            titles.append(padding(title, conf.title_seq_len, pad=word2id['<PAD>']))
            contents.append(padding(content, conf.content_seq_len, pad=word2id['<PAD>']))
    print('data size: {}'.format(len(labels)))
    np.savez_compressed(dst_path, label=labels, title=titles, content=contents)


if __name__ == '__main__':
    import os
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    emb2npz(conf.raw_emb_path, conf.emb_path)
    data2npz(conf.train_file, conf.train_data)
    data2npz(conf.dev_file, conf.dev_data)

