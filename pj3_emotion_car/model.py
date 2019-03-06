import torch
from torch import nn
import torch.nn.functional as F
import pj3_emotion_car.config as conf


def optimize_step(model, input_tensors, category_tensor, optimizer):
    model.zero_grad()
    model.train()
    output = model(input_tensors)
    loss = F.cross_entropy(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


class WordRep(nn.Module):
    def __init__(self, vocab_size, word_embed_dim):
        super(WordRep, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, word_embed_dim)

    def forward(self, input_tensors):
        sentence = input_tensors[0]
        words_embeds = self.word_embed(sentence)
        return words_embeds


class AT_LSTM(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, aspect_size):
        super(AT_LSTM, self).__init__()

        self.input_size = word_embed_dim
        self.hidden_size = conf.hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005

        self.word_rep = WordRep(vocab_size, word_embed_dim)
        # self.rnn = nn.LSTM(input_size, hidden_size)
        self.rnn_p = nn.LSTM(self.input_size, self.hidden_size // 2, bidirectional=True)

        self.AE = nn.Embedding(aspect_size, self.input_size)

        self.W_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(word_embed_dim, self.input_size)
        self.w = nn.Linear(self.hidden_size + self.input_size, 1)
        self.W_p = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x = nn.Linear(self.hidden_size, self.hidden_size)

        self.attn_softmax = nn.Softmax(dim=0)
        self.decoder_p = nn.Linear(self.hidden_size, output_size)  # TODO
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensors):
        assert len(input_tensors) == 3
        aspect_i = input_tensors[2]
        sentence = self.word_rep(input_tensors)

        length = sentence.size()[0]
        output, hidden = self.rnn_p(sentence)
        hidden = hidden[0].view(1, -1)
        output = output.view(output.size()[0], -1)

        aspect_embedding = self.AE(aspect_i)
        aspect_embedding = aspect_embedding.view(1, -1)
        aspect_embedding = aspect_embedding.expand(length, -1)
        M = F.tanh(torch.cat((self.W_h(output), self.W_v(aspect_embedding)), dim=1))
        weights = self.attn_softmax(self.w(M)).t()
        r = torch.matmul(weights, output)
        r = F.tanh(torch.add(self.W_p(r), self.W_x(hidden)))
        decoded = self.decoder_p(r)
        output = decoded
        return output


class GCAE(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, aspect_size):
        super(GCAE, self).__init__()
        self.input_size = word_embed_dim
        V = vocab_size
        D = self.input_size
        C = output_size
        A = aspect_size

        Co = 100
        Ks = [2, 3, 4]

        self.word_rep = WordRep(V, word_embed_dim)

        self.AE = nn.Embedding(A, word_embed_dim)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.fc_aspect = nn.Linear(word_embed_dim, Co)

    def forward(self, input_tensors):
        feature = self.word_rep(input_tensors)
        aspect_i = input_tensors[2]
        aspect_v = self.AE(aspect_i)  # (N, L', D)

        feature = feature.view(1, feature.size()[0], -1)

        x = [F.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        logit = self.fc1(x0)  # (N,C)
        return logit


class HEAT(nn.Module):
    def __init__(self, word_embed_dim, output_size, vocab_size, aspect_size):
        super(HEAT, self).__init__()

        self.input_size = word_embed_dim
        self.hidden_size = conf.hidden_size
        self.output_size = output_size
        self.max_length = 1
        self.lr = 0.0005

        self.word_rep = WordRep(vocab_size, word_embed_dim)

        self.rnn_a = nn.GRU(self.input_size, self.hidden_size // 2, bidirectional=True)
        self.AE = nn.Embedding(aspect_size, word_embed_dim)

        self.W_h_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v_a = nn.Linear(word_embed_dim, self.input_size)
        self.w_a = nn.Linear(self.hidden_size + word_embed_dim, 1)
        self.W_p_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.rnn_p = nn.GRU(self.input_size, self.hidden_size // 2, bidirectional=True)

        self.W_h = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_v = nn.Linear(word_embed_dim + self.hidden_size, word_embed_dim + self.hidden_size)
        self.w = nn.Linear(2 * self.hidden_size + word_embed_dim, 1)
        self.W_p = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_x = nn.Linear(self.hidden_size, self.hidden_size)

        self.decoder_p = nn.Linear(self.hidden_size + word_embed_dim, output_size)  # TODO
        self.dropout = nn.Dropout(conf.dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input_tensors):
        assert len(input_tensors) == 3
        aspect_i = input_tensors[2]
        sentence = self.word_rep(input_tensors)

        length = sentence.size()[0]
        output_a, hidden = self.rnn_a(sentence)
        output_p, _ = self.rnn_p(sentence)
        output_a = output_a.view(output_a.size()[0], -1)
        output_p = output_p.view(length, -1)
        aspect_e = self.AE(aspect_i)
        aspect_embedding = aspect_e.view(1, -1)
        aspect_embedding = aspect_embedding.expand(length, -1)
        M_a = F.tanh(torch.cat((output_a, aspect_embedding), dim=1))
        weights_a = F.softmax(self.w_a(M_a), dim=0).t()
        r_a = torch.matmul(weights_a, output_a)

        r_a_expand = r_a.expand(length, -1)
        query4PA = torch.cat((r_a_expand, aspect_embedding), dim=1)
        M_p = F.tanh(torch.cat((output_p, query4PA), dim=1))
        g_p = self.w(M_p)
        weights_p = F.softmax(g_p, dim=0).t()
        r_p = torch.matmul(weights_p, output_p)
        r = torch.cat((r_p, aspect_e), dim=1)
        decoded = self.decoder_p(r)
        ouput = decoded
        return ouput
