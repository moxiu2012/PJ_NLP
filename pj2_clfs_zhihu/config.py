
# path
raw_emb_path = r'D:\data\zhihu\ieee_zhihu_cup\word_embedding.txt'
emb_path = 'data/emb.npz'

raw_data_file = r'D:\data\zhihu\ieee_zhihu_cup\question_train_set.txt'
raw_label_file = r'D:\data\zhihu\ieee_zhihu_cup\question_topic_train_set.txt'
train_file = 'data/train_data.txt'
dev_file = 'data/dev_data.txt'

label2id_path = 'data/label2id.npz'
train_data = 'data/train_data.npz'
dev_data = 'data/dev_data.npz'

model_dict_path = 'data/model_dict_{}.pth'
model_all_path = 'data/model_all_{}.pth'

# train
epochs = 10
batch_size = 256
title_seq_len = 25
content_seq_len = 100
decay_step = 100
n_workers = 4

# model
emb_size = 256
n_classes = 50
n_layers = 3
emb_hidden_size = 250
hidden_size = 256
liner_size = 1024
lr1 = 0.01
lr2 = 0
title_dim = 1024
content_dim = 512


