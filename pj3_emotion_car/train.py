from pj3_emotion_car.model import *
from pj3_emotion_car.dataset import Dataset
import pj3_emotion_car.config as conf
import numpy as np
import torch
import tqdm
from sklearn.metrics import precision_recall_fscore_support


def score(predicted, golden):
    assert len(predicted) == len(golden)
    correct = 0
    for p, g in zip(predicted, golden):
        if p == g[0].tolist():
            correct += 1
    acc = correct / len(golden)
    predicted_all = [p[0].tolist() for p in predicted]
    golden_all = [g[0].tolist() for g in golden]
    prec, recall, f1, _ = precision_recall_fscore_support(golden_all, predicted_all, average='weighted')
    return prec, recall, f1, acc


def val(model):
    dev_data = Dataset(training=False)
    length = len(dev_data.sentences)
    model.eval()
    predicted_p = []
    for i in range(length):
        with torch.no_grad():
            input_tensor, _ = dev_data.get(i)
            output_p = model(input_tensor)
        top_n, top_i = output_p.topk(1)  # Tensor out of Variable with .data
        category_i_p = top_i.view(output_p.size()[0]).detach()
        predicted_p.append(category_i_p)
    pred_acc_p = score(predicted_p, dev_data.labels)
    print("[p:%.4f, r:%.4f, f:%.4f] acc:%.4f" % (pred_acc_p[0], pred_acc_p[1], pred_acc_p[2], pred_acc_p[3]))


def train():
    data = np.load(conf.emb_path)
    emb_mat = torch.from_numpy(data['vec'])
    aspect_emb = torch.from_numpy(data['aspect_emb'])
    word2id = data['word2id'].item()
    del data

    train_data = Dataset(training=True)
    vocab_size = len(word2id)
    polar_size = len(np.load(conf.label2id)['polar2id'].item())
    aspect_size = len(np.load(conf.label2id)['aspect2id'].item())

    Model = name_model[model_name]
    model = Model(conf.emb_size, polar_size, vocab_size, aspect_size)
    model.AE.weight = torch.nn.Parameter(aspect_emb)
    model.word_rep.word_embed.weight = nn.Parameter(emb_mat)
    # 冻结emb
    for param in model.word_rep.word_embed.parameters():
        param.requires_grad = False

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    np.random.seed([3, 1415])
    for epoch in range(1, conf.epochs + 1):
        print("# Epoch:%d" % epoch)
        loss_sum = 0

        index_list = np.arange(len(train_data.sentences))
        np.random.shuffle(index_list)
        # print(index_list)
        for idx, index in tqdm.tqdm(enumerate(index_list)):
            input_tensors, category_tensor = train_data.get(index)
            loss = optimize_step(model, input_tensors, category_tensor, optimizer)
            loss_sum += loss
            if (idx + 1) % conf.plot_every == 0:
                print('iter: {} - loss: {}'.format((idx + 1), loss_sum / (idx + 1)))

        # validation
        val(model)


if __name__ == '__main__':
    model_name = 'heat'
    name_model = {'heat': HEAT, 'gcae': GCAE, 'at_lstm': AT_LSTM}
    train()
