from __future__ import print_function, division
import torch.nn as nn
import pj4_ner_word.config as conf
import torch
from torch.autograd import Variable


def log_sum_exp(vec, m_size):
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    log_sroce = torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M
    return max_score.view(-1, m_size) + log_sroce


class CRF(nn.Module):
    def __init__(self, start_tag, end_tag, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.hidden2tag = nn.Linear(conf.word_hidden_dim, self.tagset_size * self.tagset_size)

    def cal_score(self, feats):
        """ feats (sentlen, batch_size, feature_num) : input features """
        sentlen = feats.size(0)
        batch_size = feats.size(1)
        crf_scores = self.hidden2tag(feats)
        self.crf_scores = crf_scores.view(sentlen, batch_size, self.tagset_size, self.tagset_size)
        return self.crf_scores

    def forward(self, feats, target, mask):
        """ calculate viterbi loss
            feats  (batch_size, seq_len, hidden_dim) : input features from word_rep layers
            target (batch_size, seq_len, 1) : crf label
            mask   (batch_size, seq_len) : mask for crf label
        """
        crf_scores = self.cal_score(feats)
        loss = self.get_loss(crf_scores, target, mask)
        return loss

    def get_loss(self, scores, target, mask):
        """ calculate viterbi loss
            scores (seq_len, bat_size, target_size_from, target_size_to) : class score for CRF
            target (seq_len, bat_size, 1) : crf label
            mask   (seq_len, bat_size) : mask for crf label
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)

        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()

        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, self.start_tag, :].clone()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(bat_size, self.tagset_size, 1). \
                expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = log_sum_exp(cur_values, self.tagset_size)
            mask_idx = mask[idx, :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx,
                                      cur_partition.masked_select(mask_idx))

        partition = partition[:, self.end_tag].sum()
        loss = (partition - tg_energy) / bat_size

        return loss

    def decode(self, feats, mask):
        """ decode with dynamic programming
            feats (sentlen, batch_size, feature_num) : input features
            mask (seq_len, bat_size) : mask for padding
        """
        scores = self.cal_score(feats)
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        with torch.no_grad():
            mask = Variable(1 - mask.data)
            decode_idx = Variable(torch.cuda.LongTensor(seq_len - 1, bat_size))

        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        forscores = inivalues[:, self.start_tag, :]
        back_points = list()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + forscores.contiguous().view(bat_size, self.tagset_size, 1). \
                expand(bat_size, self.tagset_size, self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size), self.end_tag)
            back_points.append(cur_bp)

        pointer = back_points[-1][:, self.end_tag]
        decode_idx[-1] = pointer
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(bat_size, 1))
            decode_idx[idx] = pointer.view(-1)
        return decode_idx


class WORD_REP(nn.Module):

    def __init__(self, char_size, vocab_size, pre_word_embeddings):
        super(WORD_REP, self).__init__()
        self.char_size = char_size
        self.word_size = vocab_size

        self.char_embeds = nn.Embedding(char_size, conf.char_embedding_dim)
        self.word_embeds = nn.Embedding(vocab_size, conf.word_embedding_dim)

        self.cnn = nn.Conv2d(1, conf.cnn_filter_num, (3, conf.char_embedding_dim), padding=(2, 0))
        self.word_lstm_cnn = nn.LSTM(conf.word_embedding_dim + conf.cnn_filter_num, conf.word_hidden_dim // 2,
                                     num_layers=conf.word_lstm_layers, bidirectional=True,
                                     dropout=conf.dropout_ratio)
        self.dropout = nn.Dropout(p=conf.dropout_ratio)
        self.batch_size = 1
        self.word_seq_length = 1
        self.word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def set_batch_seq_size(self, sentence):
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def forward(self, word_seq, cnn_features):
        self.set_batch_seq_size(word_seq)
        cnn_features = cnn_features.view(cnn_features.size(0) * cnn_features.size(1), -1)
        cnn_features = self.char_embeds(cnn_features).view(cnn_features.size(0), 1, cnn_features.size(1), -1)
        cnn_features = self.cnn(cnn_features)
        d_char_out = nn.functional.max_pool2d(cnn_features, kernel_size=(cnn_features.size(2), 1))
        d_char_out = d_char_out.view(self.word_seq_length, self.batch_size, conf.cnn_filter_num)
        word_emb = self.word_embeds(word_seq)

        word_input = torch.cat((word_emb, d_char_out), dim=2)
        word_input = self.dropout(word_input)

        lstm_out, _ = self.word_lstm_cnn(word_input)
        lstm_out = self.dropout(lstm_out)

        return lstm_out


class ner_model(nn.Module):

    def __init__(self, vocab_size, char_size, crf_start_tag, crf_end_tag, crf_target_size, embedding_tensor):
        super(ner_model, self).__init__()

        self.word_rep = WORD_REP(char_size, vocab_size, embedding_tensor)

        self.crf = CRF(crf_start_tag, crf_end_tag, crf_target_size)

    def forward(self, word_seq, cnn_features, crf_target, crf_mask):
        word_representations = self.word_rep(word_seq, cnn_features)
        loss_crf = self.crf(word_representations, crf_target, crf_mask)
        loss = loss_crf
        return loss
