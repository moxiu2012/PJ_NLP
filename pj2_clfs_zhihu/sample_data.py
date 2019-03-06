from collections import Counter
import pj2_clfs_zhihu.config as conf
import numpy as np
from sklearn.model_selection import train_test_split


def contain(labels, ids):
    for id in labels:
        if id not in ids:
            return False
    return True


def more(datas, data_id, labels, ids):
    label_ids = labels.copy()
    for id in labels:
        if id in ids:
            if ids[id] < 0:
                continue
            ids[id] -= 1
        else:
            label_ids.remove(id)
    if len(label_ids) > 0:
        datas.append('{}\t{}'.format(data_id, ','.join(label_ids)))


def count_ids(src_path):
    ids = []
    with open(src_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            ids.extend(line.replace('\n', '').split('\t')[1].split(','))
        print('src data size: {}'.format(idx + 1))
    return Counter(ids).most_common()


def find_data(ids, first=True):
    datas = []
    with open(conf.raw_label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data_id, labels = line.replace('\n', '').split('\t')
            labels = labels.split(',')
            keep = contain(labels, ids)
            if keep:
                datas.append(line.replace('\n', ''))

            if not keep and not first:
                more(datas, data_id, labels, ids)

    labels = []
    [labels.extend(line.split('\t')[1].split(',')) for line in datas]
    id_counts = Counter(labels).most_common()
    return datas, id_counts


def write(datas, path):
    with open(path, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(datas))


def filter(lines, id_counts, threshold):
    print('threshold: {}'.format(threshold))
    ids = {line[0]: line[1] - threshold for line in id_counts if line[1] - threshold > 0}
    datas = []
    for line in lines:
        labels = line.split('\t')[1].split(',')
        if len(labels) == 1 and labels[0] in ids:
            if ids[labels[0]] <= 0:
                datas.append(line)
            else:
                ids[labels[0]] -= 1
        else:
            datas.append(line)
    return datas


def choice_data(top_start, top_end):
    # 统计原有标签集数据中每类标签分布
    id_counts = count_ids(conf.raw_label_file)
    # 选出子集标签
    ids = [line[0] for line in id_counts][top_start:top_end]
    # 寻找标签都在子集的数据，统计这部分数据的标签分布
    _, id_counts = find_data(ids, first=True)

    # 寻找子标签都在子集的数据和 某些子标签在子集的数据并删去不在子集的子标签
    id_dict = {line[0]: id_counts[0][1] - line[1] for line in id_counts}
    datas, id_counts = find_data(id_dict, first=False)

    # 过滤一部分数据使数据分布比较平衡
    threshold = int((id_counts[0][1] - id_counts[-1][1]) * 0.4 + id_counts[-1][1])
    # threshold = 10000
    label_datas = filter(datas, id_counts, threshold)

    # 根据标签 筛选训练数据
    data_ids = set([line.split('\t')[0] for line in label_datas])
    train_set = []
    with open(conf.raw_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '')
            data_id = line.split('\t')[0]
            if data_id in data_ids:
                train_set.append(line)

    data_label_id, label2id = get_data_label_id(label_datas)
    extract_data_and_split(train_set, data_label_id, label2id)


def get_data_label_id(label_datas):
    """ 获取数据对应的标签id映射字典 """
    label2id = {}
    data_label_id = {}
    for line in label_datas:
        line = line.replace('\n', '')
        data_id = line.split('\t')[0]
        label_ids = line.split('\t')[1].split(',')

        for label_id in label_ids:
            label2id[label_id] = label2id.get(label_id, len(label2id))

        label_ids = [label2id[label] for label in label_ids]
        data_label_id[data_id] = label_ids

    return data_label_id, label2id


def extract_data_and_split(train_set, data_label_id, label2id):
    """1、处理数据和标签, 抽取title_word和content_word，将label和data对应起来 保存到文件中
        2、划分数据集"""
    datas = []
    for line in train_set:
        data_id, _, title_word, _, content_word = line.replace('\n', '').split('\t')
        labels = ','.join([str(label) for label in data_label_id[data_id]])
        info = '{}\t{}\t{}'.format(labels, title_word, content_word)
        datas.append(info)

    train_data, val_data = train_test_split(datas, test_size=0.05, random_state=2019)
    print('label num: {} - data num:{}'.format(len(label2id), len(datas)))
    print("train num: {} - dev num: {}".format(len(train_data), len(val_data)))

    np.savez_compressed(conf.label2id_path, data_label_id=data_label_id, label2id=label2id)
    with open(conf.train_file, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(train_data))
    with open(conf.dev_file, 'w', encoding='utf-8') as fw:
        fw.write('\n'.join(val_data))


if __name__ == '__main__':
    choice_data(50, 100)


