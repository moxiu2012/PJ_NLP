import pj1_clf_news.config as conf
from keras.utils import to_categorical
import numpy as np


class Dataset:
    def __init__(self, training=True):
        data = np.load(conf.emb_path)
        self.emb_mat = data['emb_mat']
        self.word2id = data['word2id'].item()
        self.vocab_size = len(self.word2id)
        self.path = conf.train_data if training else conf.dev_data
        self.steps_per_epoch = self.get_steps_per_epoch()
        self.label2id = np.load(conf.id_vocab_path)['label2id'].item()
        self.tags = []
        del data

    def padding(self, text, max_len):
        texts = text.split()
        texts = texts + ['<PAD>'] * (max_len - len(texts)) if len(texts) < max_len else texts[:max_len]
        return texts

    def get_steps_per_epoch(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for num, _ in enumerate(f):
                pass
        return num // conf.batch_size + 1  # 函数式模型必须要设置steps_per_epoch

    def gen_batch_data(self):
        t_id_all, c_id_all, labels = [], [], []
        step = 0
        while True:
            with open(self.path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.rstrip()
                    label, title, content = line.split('\t')
                    # padding
                    titles = self.padding(title, conf.title_seq_len)
                    contents = self.padding(content, conf.content_seq_len)
                    # word2id
                    t_id_all.append([self.word2id[word] if word in self.word2id else self.word2id['<UNK>']
                                     for word in titles])
                    c_id_all.append([self.word2id[word] if word in self.word2id else self.word2id['<UNK>']
                                     for word in contents])
                    # label2id
                    labels.append(self.label2id[label])

                    if len(labels) == conf.batch_size:
                        step += 1
                        if step <= self.steps_per_epoch:
                            self.tags.extend(labels)
                        input_1 = np.array(t_id_all[: conf.batch_size])
                        input_2 = np.array(c_id_all[: conf.batch_size])
                        output = np.array(to_categorical(labels[: conf.batch_size], conf.n_classes))
                        t_id_all, c_id_all, labels = [], [], []
                        yield [input_1, input_2], output
