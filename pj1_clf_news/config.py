
# path
raw_emb_path = r'D:\data\embedding\merge_sgns_bigram_char300.txt'
emb_path = r'data/emb.npz'

raw_data_path = r'data/news.txt'
data_file = 'data/data.txt'

train_data = 'data/train_data.txt'
dev_data = 'data/dev_data.txt'
id_vocab_path = 'data/id_vocab.npz'

model_path = 'data/{}.h5'

# train

batch_size = 64
epochs = 10
test_size = 0.2

# model
n_classes = 18
content_seq_len = 300
title_seq_len = 15
emb_size = 300
emb_hidden_size = 300