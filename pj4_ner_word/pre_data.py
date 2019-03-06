from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pickle
import pj4_ner_word.config as conf
import functools
from torch.utils.data.dataset import Dataset
import os


class CRFDataset(Dataset):
    """ Dataset Class for ner model """
    def __init__(self, word_tensor, label_tensor, mask_tensor, len_tensor, cnn_features):
        self.word_tensor = word_tensor
        self.label_tensor = label_tensor
        self.mask_tensor = mask_tensor
        self.len_tensor = len_tensor
        self.cnn_features = cnn_features

    def __getitem__(self, index):
        return self.word_tensor[index], self.label_tensor[index], self.mask_tensor[index], \
               self.len_tensor[index], self.cnn_features[index]

    def __len__(self):
        return self.word_tensor.size(0)


def iob_iobes(tags):
    """  IOB -> IOBES """
    iob2(tags)
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iob2(tags):
    """ Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2. """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True


def get_data_label(lines):
    datas = [[words.split('|')[0] for words in line.split(' ')] for line in lines]
    labels = [iob_iobes([words.split('|')[1] for words in line.split(' ')]) for line in lines]
    return datas, labels


def get_word_char_map(train_datas):
    words = set([word for line in train_datas for word in line])
    word2id = {word: idx + 1 for idx, word in enumerate(words)}
    word2id['<unk>'] = 0
    word2id['<eof>'] = len(word2id)

    chars = set([char_ for line in train_datas for word in line for char_ in word])
    char2id = {char_: idx for idx, char_ in enumerate(chars)}
    char2id['<u>'] = len(char2id)
    char2id[' '] = len(char2id)
    char2id['\n'] = len(char2id)
    return word2id, char2id


def read_data():
    data = [line.replace('\n', '') for line in open(conf.raw_data_path, 'r', encoding='utf-8')]
    train_data, dev_data = train_test_split(data, test_size=0.2, random_state=2019)

    all_words = set([word.split('|')[0] for line in data for word in line.split(' ')])
    dev_datas, dev_labels = get_data_label(dev_data)
    train_datas, train_labels = get_data_label(train_data)
    return train_datas, train_labels, dev_datas, dev_labels, all_words


def load_embedding(emb_file, word2id, all_words):
    in_doc_freq_num = len(word2id)
    rand_embedding_tensor = torch.FloatTensor(in_doc_freq_num, conf.word_embedding_dim)
    bias = np.sqrt(3.0 / rand_embedding_tensor.size(1))
    nn.init.uniform_(rand_embedding_tensor, -bias, bias)

    indoc_embedding_array = list()
    indoc_word_array = list()
    for line in open(emb_file, 'r', encoding='utf-8'):
        line = line.split(' ')
        vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if line[0] == 'unk':
            rand_embedding_tensor[0] = torch.FloatTensor(vector)  # unk is 0
        elif line[0] in word2id:
            rand_embedding_tensor[word2id[line[0]]] = torch.FloatTensor(vector)
        elif line[0] in all_words:
            indoc_embedding_array.append(vector)
            indoc_word_array.append(line[0])

    embedding_tensor_0 = torch.FloatTensor(np.asarray(indoc_embedding_array))
    embedding_tensor = torch.cat([rand_embedding_tensor, embedding_tensor_0], 0)

    for word in indoc_word_array:
        word2id[word] = len(word2id)
    return word2id, embedding_tensor


def calc_threshold_mean(features):
    """ calculate the threshold for bucket by mean """
    lines_len = list(map(lambda t: len(t) + 1, features))
    average = int(sum(lines_len) / len(lines_len))
    lower_line = list(filter(lambda t: t < average, lines_len))
    upper_line = list(filter(lambda t: t >= average, lines_len))
    lower_average = int(sum(lower_line) / len(lower_line))
    upper_average = int(sum(upper_line) / len(upper_line))
    max_len = max(lines_len)
    return [lower_average, average, upper_average, max_len]


def construct_bucket_mean_vb_wc(word_features, input_label, label_dict, char_dict, word_dict):
    """Construct bucket by mean for viterbi decode, word-level and char-level"""
    labels = list(map(lambda t: list(map(lambda m: label_dict[m], t)), input_label))
    char_features = [list(map(lambda m: list(map(lambda t: char_dict.get(t, char_dict['<u>']), m)), line))
                     for line in word_features]

    fea_len = [list(map(lambda t: len(t) + 1, f)) for f in char_features]
    forw_features = [[char_dict[' ']] + list(functools.reduce(lambda x, y: x + [char_dict[' ']] + y, sentence))
                     + [char_dict['\n']] for sentence in char_features]

    new_labels = list(map(lambda t: [label_dict['<start>']] + list(t), labels))
    thresholds = calc_threshold_mean(fea_len)
    new_word_features = list(map(lambda t: list(map(lambda x: x.lower(), t)), word_features))
    new_word_features = list(
        map(lambda t: list(map(lambda m: word_dict.get(m, word_dict['<unk>']), t)), new_word_features))

    dataset = construct_bucket_vb_wc(new_word_features, forw_features, fea_len, new_labels,
                                     char_features, thresholds, word_dict['<eof>'], char_dict['\n'],
                                     label_dict['<pad>'], len(label_dict))
    return dataset


def construct_bucket_vb_wc(word_features, forw_features, fea_len, input_labels, char_features, thresholds,
                           pad_word_feature, pad_char_feature, pad_label, label_size):
    """Construct bucket by thresholds for viterbi decode, word-level and char-level"""
    word_max_len = max([len(c) for c_fs in char_features for c in c_fs])
    buckets = [[[], [], [], [], []] for _ in range(len(thresholds))]
    for f_f, f_l, w_f, i_l, c_f in zip(forw_features, fea_len, word_features, input_labels, char_features):
        cur_len = len(f_l)
        idx = 0
        cur_len_1 = cur_len + 1
        w_l = max(f_l) - 1

        while thresholds[idx] < cur_len_1:
            idx += 1
        buckets[idx][0].append(w_f + [pad_word_feature] * (thresholds[idx] - cur_len))  # word
        buckets[idx][1].append([i_l[ind] * label_size + i_l[ind + 1] for ind in range(0, cur_len)] + [
            i_l[cur_len] * label_size + pad_label] + [pad_label * label_size + pad_label] * (
                                       thresholds[idx] - cur_len_1))  # has additional start, label
        buckets[idx][2].append([1] * cur_len_1 + [0] * (thresholds[idx] - cur_len_1))  # has additional start, mask
        buckets[idx][3].append([len(f_f) + thresholds[idx] - len(f_l), cur_len_1, w_l])
        buckets[idx][4].append(
            [c + [pad_char_feature] * (word_max_len - len(c)) for c in c_f] + [[pad_char_feature] * word_max_len] * (
                    thresholds[idx] - cur_len))
    bucket_dataset = [CRFDataset(torch.LongTensor(bucket[0]), torch.LongTensor(bucket[1]), torch.ByteTensor(bucket[2]),
                                 torch.LongTensor(bucket[3]), torch.LongTensor(bucket[4])) for bucket in buckets]
    return bucket_dataset


def process(emb_reset=True):
    # get_data
    train_datas, train_labels, dev_datas, dev_labels, all_words = read_data()
    # to_id
    word2id, char2id = get_word_char_map(train_datas)
    labels = set([label for line in train_labels + dev_labels for label in line])
    label2id = {label: idx for idx, label in enumerate(labels)}
    label2id['<start>'] = len(label2id)
    label2id['<pad>'] = len(label2id)

    print('train_datas size: {} - dev_datas size: {}'.format(len(train_datas), len(dev_datas)))
    print('label2id size: {} - char2id size: {} - char2id size: {}'.format(len(label2id), len(char2id), len(word2id)))

    # embedding_tensor
    if os.path.exists(conf.data_path) and not emb_reset:
        with open(conf.data_path, 'rb') as f:
            datas = pickle.load(f)
            embedding_tensor = datas['embedding_tensor']
            word2id = datas['f_map']
            del datas
    else:
        word2id, embedding_tensor = load_embedding(conf.raw_emb_file, word2id, all_words)
    print('word2id: ', len(word2id))
    print('embedding_tensor: ', embedding_tensor.size())

    # packed_dataset
    dataset = construct_bucket_mean_vb_wc(train_datas, train_labels, label2id, char2id, word2id)
    dev_dataset = construct_bucket_mean_vb_wc(dev_datas, dev_labels, label2id, char2id, word2id)
    dataset_loader = [torch.utils.data.DataLoader(tup, conf.batch_size, shuffle=True, drop_last=False)
                      for tup in dataset]
    dev_dataset_loader = [torch.utils.data.DataLoader(tup, conf.batch_size, shuffle=False, drop_last=False)
                          for tup in dev_dataset]

    with open(conf.data_path, 'wb') as f:
        datas = {
            'dataset_loader': dataset_loader,
            'dev_dataset_loader': dev_dataset_loader,
            'CRF_l_map': label2id,
            'c_map': char2id,
            'f_map': word2id,
            'embedding_tensor': embedding_tensor,
        }
        pickle.dump(datas, f)


if __name__ == '__main__':
    import os
    data_dir = 'data/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    process(False)
