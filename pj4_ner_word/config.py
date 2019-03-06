
# path
raw_emb_file = r'D:\data\embedding\merge_embedding_300.txt'
raw_data_path = 'data/ner_corpus.txt'
data_path = 'data/datas.pkl'
model_path = 'data/model.pth'

# train
batch_size = 16
epochs = 20
early_stop = 5

# model
char_lstm_hidden_dim = 300
word_hidden_dim = 300
word_embedding_dim = 300
char_embedding_dim = 30
cnn_filter_num = 30
word_lstm_layers = 1
dropout_ratio = 0.55

lr = 0.015
lr_decay = 0.05
momentum = 0.9
clip_grad = 5.0

