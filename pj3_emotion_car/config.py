
# path
attr_path = 'data/attribute.json'
polar_path = 'data/polarity.json'
raw_data_path = 'data/car_emtion_corpus.csv'
raw_emb_path = r'D:\data\embedding\merge_embedding.npz'
emb_path = 'data/emb.npz'

train_data = 'data/train.npz'
dev_data = 'data/dev.npz'
label2id = 'data/label2id.npz'

# train
epochs = 5
hidden_size = 128
plot_every = 2000

# model
optimizer = 'Adam'
emb_size = 300
aspect_size = 10
lr = 0.2
dropout = 0.5

