from torch import nn
import pj2_clfs_zhihu.config as conf
import torch as t
import torch.nn.functional as F


def get_params_num(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))


class Cnn(nn.Module):

    def __init__(self, vocab_size, emb_mat):
        kernel_size = 3
        filters = 128
        super(Cnn, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.encoder = nn.Embedding(vocab_size, conf.emb_size)
        self.title_conv = nn.Sequential(
            nn.Conv1d(conf.emb_size, filters, kernel_size=kernel_size),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(conf.title_seq_len - 2 * kernel_size + 2))
        )
        self.content_conv = nn.Sequential(
            nn.Conv1d(conf.emb_size, filters, kernel_size=3),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(conf.content_seq_len - 2 * kernel_size + 2))
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear((filters + filters), 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, conf.n_classes),
        )

        self.encoder.weight.data.copy_(emb_mat)

    def forward(self, title, content):
        title = self.encoder(title)
        title = title.permute(0, 2, 1)
        content = self.encoder(content)
        content = content.permute(0, 2, 1)
        title_out = self.title_conv(title)
        content_out = self.content_conv(content)
        conv_out = t.cat((title_out, content_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits

    def get_optimizer(self, lr1=0.001, lr2=0):
        # 分层设置学习率
        encoder_parmas = self.encoder.parameters()
        orther_params = [param for name, param in self.named_parameters() if name.find('encoder') == -1]
        opt = t.optim.Adam([dict(params=orther_params, lr=lr1), dict(params=encoder_parmas, lr=lr2)])
        return opt


class TextCNN(t.nn.Module):
    """ MultiCNNTextBNDeep神经网络模型 """

    def __init__(self, vocab_size, emb_mat):
        kernel_sizes = [1, 2, 3, 4]
        super(TextCNN, self).__init__()
        self.encoder = nn.Embedding(vocab_size, conf.emb_size)
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

        title_convs = [nn.Sequential(
            nn.Conv1d(in_channels=conf.emb_size, out_channels=conf.emb_hidden_size, kernel_size=kernel_size),
            nn.BatchNorm1d(conf.emb_hidden_size),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=conf.emb_hidden_size, out_channels=conf.emb_hidden_size, kernel_size=kernel_size),
            nn.BatchNorm1d(conf.emb_hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(conf.title_seq_len - kernel_size * 2 + 2))
        )
            for kernel_size in kernel_sizes]

        content_convs = [nn.Sequential(
            nn.Conv1d(in_channels=conf.emb_size, out_channels=conf.emb_hidden_size, kernel_size=kernel_size),
            nn.BatchNorm1d(conf.emb_hidden_size),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=conf.emb_hidden_size, out_channels=conf.emb_hidden_size, kernel_size=kernel_size),
            nn.BatchNorm1d(conf.emb_hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(conf.content_seq_len - kernel_size * 2 + 2))
        )
            for kernel_size in kernel_sizes]

        self.title_convs = nn.ModuleList(title_convs)
        self.content_convs = nn.ModuleList(content_convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (conf.emb_hidden_size + conf.emb_hidden_size), conf.liner_size),
            nn.BatchNorm1d(conf.liner_size),
            nn.ReLU(inplace=True),
            nn.Linear(conf.liner_size, conf.n_classes)
        )

        self.encoder.weight.data.copy_(emb_mat)

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)

        title_out = [title_conv(title.permute(0, 2, 1)) for title_conv in self.title_convs]
        content_out = [content_conv(content.permute(0, 2, 1)) for content_conv in self.content_convs]
        conv_out = t.cat((title_out + content_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits

    def get_optimizer(self, lr1, lr2=0, weight_decay=0):
        if lr2 is None:
            lr2 = lr1 * 0.5
        encoder_params = self.encoder.parameters()
        other_params = [param_ for name_, param_ in self.named_parameters() if name_.find('encoder') == -1]
        optimizer = t.optim.Adam([
        # optimizer = t.optim.SGD([
            dict(params=other_params, weight_decay=weight_decay, lr=lr1),
            dict(params=encoder_params, lr=lr2)])
        return optimizer


class CnnRes(nn.Module):

    def __init__(self, vocab_size, emb_mat):
        kernel_size = 3
        filters = 128
        super(CnnRes, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.encoder = nn.Embedding(vocab_size, conf.emb_size)

        self.title_conv1 = nn.Sequential(
            nn.Conv1d(conf.emb_size, filters, kernel_size=kernel_size, ),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, conf.emb_size, kernel_size=kernel_size),
            nn.BatchNorm1d(conf.emb_size),
        )

        self.title_conv2 = nn.Sequential(
            nn.Conv1d(conf.emb_size, filters * 2, kernel_size=kernel_size),
            nn.BatchNorm1d(filters * 2),
        )
        self.title_conv3 = nn.Sequential(
            nn.Conv1d(filters * 2, filters * 4, kernel_size=kernel_size),
            nn.BatchNorm1d(filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(conf.title_seq_len - 2 * kernel_size + 2))
        )

        self.content_conv1 = nn.Sequential(
            nn.Conv1d(conf.emb_size, filters, kernel_size=kernel_size),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.Conv1d(filters, conf.emb_size, kernel_size=kernel_size),
            nn.BatchNorm1d(conf.emb_size),
        )

        self.content_conv2 = nn.Sequential(
            nn.Conv1d(conf.emb_size, filters * 2, kernel_size=kernel_size),
            nn.BatchNorm1d(filters * 2),
        )
        self.content_conv3 = nn.Sequential(
            nn.Conv1d(filters * 2, filters * 4, kernel_size=kernel_size),
            nn.BatchNorm1d(filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(conf.content_seq_len - 2 * kernel_size + 2))
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, conf.n_classes),
        )

        self.encoder.weight.data.copy_(emb_mat)

    def forward(self, title, content):
        title_em = self.encoder(title)
        title_1 = title_em.permute(0, 2, 1)
        title_2 = self.title_conv1(title_1)
        title_2 = t.cat((title_1, title_2), dim=2)
        title_2 = F.relu(title_2)
        title_3 = self.title_conv2(title_2)
        title_3 = t.cat((title_2, title_3), dim=2)
        title_3 = F.relu(title_3)
        title_out = self.title_conv3(title_3)

        content_em = self.encoder(content)
        content_1 = content_em.permute(0, 2, 1)
        content_2 = self.content_conv1(content_1)
        content_2 = t.cat((content_1, content_2), dim=2)
        content_2 = F.relu(content_2)
        content_3 = self.content_conv2(content_2)
        content_3 = t.cat((content_2, content_3), dim=2)
        content_3 = F.relu(content_3)
        content_out = self.content_conv3(content_3)

        conv_out = t.cat((title_out, content_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits

    def get_optimizer(self, lr1=0.001, lr2=0):
        # 分层设置学习率
        encoder_parmas = self.encoder.parameters()
        orther_params = [param for name, param in self.named_parameters() if name.find('encoder') == -1]
        opt = t.optim.Adam([dict(params=orther_params, lr=lr1), dict(params=encoder_parmas, lr=lr2)])
        return opt


def kmax_pool(data, dim, k_max):
    """拿出最大的k个值 并且保留原有顺序"""
    index = data.topk(k_max, dim=dim)[1].sort(dim=dim)[0]
    return data.gather(dim, index)


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_mat):
        self.k_max = 3
        super(LSTM, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.encoder = nn.Embedding(vocab_size, conf.emb_size)

        self.title_lstm = nn.LSTM(conf.emb_size, conf.hidden_size, num_layers=3, bidirectional=True)
        self.content_lstm = nn.LSTM(conf.emb_size, conf.hidden_size, num_layers=3, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(conf.hidden_size * 4 * self.k_max, conf.liner_size),
            nn.BatchNorm1d(conf.liner_size),
            nn.ReLU(inplace=True),
            nn.Linear(conf.liner_size, conf.n_classes)
        )
        self.encoder.weight.data.copy_(emb_mat)

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
        content_out = self.content_lstm(content.permute(1, 0, 2))[0].permute(1, 2, 0)
        # 最大取样
        title_max = kmax_pool(title_out, 2, self.k_max)
        content_max = kmax_pool(content_out, 2, self.k_max)
        out = t.cat((title_max, content_max), dim=1)
        reshape = out.view(out.size(0), -1)
        logits = self.fc(reshape)
        return logits

    def get_optimizer(self, lr1=0.001, lr2=0):
        # 分层设置学习率
        encoder_parmas = self.encoder.parameters()
        orther_params = [param for name, param in self.named_parameters() if name.find('encoder') == -1]
        opt = t.optim.Adam([dict(params=orther_params, lr=lr1), dict(params=encoder_parmas, lr=lr2)])
        return opt


class RCNN(nn.Module):
    def __init__(self, vocab_size, emb_mat):
        super(RCNN, self).__init__()
        self.k_max = 3
        self.loss_fn = nn.MultiLabelSoftMarginLoss()
        self.encoder = nn.Embedding(vocab_size, conf.emb_size)
        self.encoder.weight.data.copy_(emb_mat)

        self.title_lstm = nn.LSTM(conf.emb_size, conf.hidden_size,
                                  bidirectional=True, num_layers=conf.n_layers)
        self.title_conv = nn.Sequential(
            nn.Conv1d(conf.hidden_size * 2 + conf.emb_size, conf.title_dim, kernel_size=3),
            nn.BatchNorm1d(conf.title_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(conf.title_dim, conf.title_dim, kernel_size=3),
            nn.BatchNorm1d(conf.title_dim),
            nn.ReLU(inplace=True)
        )

        self.content_lstm = nn.LSTM(conf.emb_size, conf.hidden_size,
                                    bidirectional=True, num_layers=conf.n_layers)
        self.content_conv = nn.Sequential(
            nn.Conv1d(conf.hidden_size * 2 + conf.emb_size, conf.content_dim, kernel_size=3),
            nn.BatchNorm1d(conf.content_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(conf.content_dim, conf.content_dim, kernel_size=3),
            nn.BatchNorm1d(conf.content_dim),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.k_max * (conf.title_dim + conf.content_dim), conf.liner_size),
            nn.BatchNorm1d(conf.liner_size),
            nn.ReLU(inplace=True),
            nn.Linear(conf.liner_size, conf.n_classes)
        )

    def forward(self, title, content):
        title_in = self.encoder(title)
        title_lstm = self.title_lstm(title_in.permute(1, 0, 2))[0].permute(1, 2, 0)
        title_em = title_in.permute(0, 2, 1)
        title_lstm = t.cat((title_em, title_lstm), dim=1)
        title_out = kmax_pool(self.title_conv(title_lstm), 2, self.k_max)

        content_in = self.encoder(content)
        content_lstm = self.content_lstm(content_in.permute(1, 0, 2))[0].permute(1, 2, 0)
        content_em = content_in.permute(0, 2, 1)
        content_lstm = t.cat((content_em, content_lstm), dim=1)
        content_out = kmax_pool(self.content_conv(content_lstm), 2, self.k_max)

        out = t.cat((title_out, content_out), dim=1)
        reshaped = out.view(out.size(0), -1)
        logits = self.fc(reshaped)
        return logits

    def get_optimizer(self, lr1=0.001, lr2=0):
        # 分层设置学习率
        encoder_parmas = self.encoder.parameters()
        orther_params = [param for name, param in self.named_parameters() if name.find('encoder') == -1]
        opt = t.optim.Adam([dict(params=orther_params, lr=lr1), dict(params=encoder_parmas, lr=lr2)])
        return opt


class FastText(nn.Module):
    def __init__(self, vocab_size, emb_mat):
        super(FastText, self).__init__()
        self.loss_fn = nn.MultiLabelSoftMarginLoss()

        self.encoder = nn.Embedding(vocab_size, conf.emb_size)
        self.encoder.weight.data.copy_(emb_mat)

        self.pre1 = nn.Sequential(
            nn.Linear(conf.emb_size, 2*conf.emb_size),
            nn.BatchNorm1d(2*conf.emb_size),
            nn.ReLU(inplace=True)
        )

        self.pre2 = nn.Sequential(
            nn.Linear(conf.emb_size, 2 * conf.emb_size),
            nn.BatchNorm1d(2 * conf.emb_size),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(4*conf.emb_size, conf.liner_size),
            nn.BatchNorm1d(conf.liner_size),
            nn.ReLU(inplace=True),
            nn.Linear(conf.liner_size, conf.n_classes)
        )

    def forward(self, title, content):
        title_em = self.encoder(title)
        content_em = self.encoder(content)
        title_shape = title_em.size()
        content_shape = content_em.size()
        title_reshape = title_em.view(-1, conf.emb_size)
        content_reshape = content_em.view(-1, conf.emb_size)
        title_pre = self.pre1(title_reshape).view(title_shape[0], title_shape[1], -1)
        content_pre = self.pre2(content_reshape).view(content_shape[0], content_shape[1], -1)
        title_ = t.mean(title_pre, dim=1)
        content_ = t.mean(content_pre, dim=1)
        out = t.cat((title_, content_), dim=1).view(title_shape[0], -1)
        logits = self.fc(out)
        return logits

    def get_optimizer(self, lr1=0.001, lr2=0):
        # 分层设置学习率
        encoder_parmas = self.encoder.parameters()
        orther_params = [param for name, param in self.named_parameters() if name.find('encoder') == -1]
        opt = t.optim.Adam([dict(params=orther_params, lr=lr1), dict(params=encoder_parmas, lr=lr2)])
        return opt
