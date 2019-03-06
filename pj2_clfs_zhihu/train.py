from torch.utils.data.dataloader import DataLoader
from pj2_clfs_zhihu.dataset import ZhiHuData
from pj2_clfs_zhihu.model import *
import pj2_clfs_zhihu.config as conf
import numpy as np
from torch.autograd import Variable
import torch as t
import tqdm
from tensorboardX import SummaryWriter
import math


writer = SummaryWriter(comment='zhihu')


def calc_f1(predict_and_true_labels):
    right_label_num = 0  # 总命中标签数量
    all_marked_label_num = 0  # 总标签数量
    all_pred_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_and_true_labels:
        marked_label_set = set(marked_labels)
        pred_label_set = set(predict_labels)
        right_label_set = marked_label_set & pred_label_set
        all_marked_label_num += len(marked_label_set)
        all_pred_label_num += len(pred_label_set)
        right_label_num += len(right_label_set)

    precision = right_label_num / (all_pred_label_num + 1e-10)
    recall = right_label_num / all_marked_label_num
    f1 = 2*precision*recall / (precision + recall + 1e-10)
    return f1, precision, recall


def calc_score(predict_and_true_labels, topk=5):
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0] * topk  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_and_true_labels:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), topk)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, topk), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num

    return (precision * recall) / (precision + recall + 1e-10), precision, recall


def val(model):
    """ 计算模型在验证集上的分数 """
    top_k = 3
    # 状态置为验证
    model.eval()

    # 数据准备
    dataset = ZhiHuData(conf.dev_data)
    data_loader = DataLoader(dataset, batch_size=conf.batch_size)
    # 预测
    predict_label_and_marked_label_list = []
    for i, batch in enumerate(data_loader):
        title, content, label = batch
        with t.no_grad():
            title, content = Variable(title.cuda()), Variable(content.cuda())
        score = model(title, content)
        pred_value = score.data.topk(top_k, dim=1)[0].cpu()
        pred_index = score.data.topk(top_k, dim=1)[1].cpu()
        # 计算得分
        true_value = label.data.float().topk(top_k, dim=1)[0]
        true_index = label.data.float().topk(top_k, dim=1)[1]
        tmp = []
        for jj in range(label.size(0)):
            true = true_index[jj][true_value[jj] > 0]
            pred = pred_index[jj][pred_value[jj] > 0]
            tmp.append((pred.tolist(), true.tolist()))
        predict_label_and_marked_label_list.extend(tmp)
    scores, prec_, recall_ = calc_score(predict_label_and_marked_label_list, topk=top_k)
    print('calc_score score: {} - prec: {} - recall: {}'.format(scores, prec_, recall_))
    scores, prec_, recall_ = calc_f1(predict_label_and_marked_label_list)
    print('calc_f1 score: {} - prec: {} - recall: {}'.format(scores, prec_, recall_))

    # 状态置为训练
    model.train()
    return scores, prec_, recall_


def train():
    data = np.load(conf.emb_dict_path)
    emb_mat = t.from_numpy(data['vec'])
    word2id = data['word2id'].item()
    del data
    vocab_size = len(word2id)
    print('vocab size : {}'.format(vocab_size))

    dataset = ZhiHuData(conf.train_data)
    data_loader = DataLoader(dataset=dataset, batch_size=conf.batch_size)

    Model = name_model[model_name]
    model = Model(vocab_size, emb_mat).cuda()

    # 打印参数
    get_params_num(model)
    optimizer = model.get_optimizer(conf.lr1, conf.lr2)
    best_score = 0
    step = 0
    for epoch in range(conf.epochs):
        print('epoch:===>', epoch)
        for i, batch in tqdm.tqdm(enumerate(data_loader)):
            title, content, label = batch
            title, content, label = Variable(title.cuda()), Variable(content.cuda()), Variable(label.cuda())
            optimizer.zero_grad()
            output = model(title, content)
            loss = model.loss_fn(output, label.float())
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('train loss', loss, step)

        scores, prec_, recall_ = val(model)

        if best_score < scores:
            best_score = scores
            t.save(model, conf.model_all_path.format(model_name))
            # t.save(model.state_dict(), conf.model_dict_path.format(model_name))
    # 可视化
    writer.add_graph(model, (title, content))
    writer.close()


if __name__ == '__main__':
    name_model = {'cnn': Cnn, 'cnn_res': CnnRes, 'text_cnn': TextCNN, 'lstm': LSTM, 'rcnn': RCNN,
                  'fasttext': FastText}
    model_name = 'text_cnn'

    train()
