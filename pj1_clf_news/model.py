from keras.layers import *
import pj1_clf_news.config as conf
from keras.models import Model


def model_lstm(vocab_size, n_classes, emb_mat):
    # model
    title_in = Input(shape=(conf.title_seq_len,), dtype='int32')
    title_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(title_in)
    title_x = LSTM(conf.emb_hidden_size)(title_x)

    content_in = Input(shape=(conf.content_seq_len,), dtype='int32')
    content_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(content_in)
    content_x = LSTM(conf.emb_hidden_size)(content_x)

    x = add([title_x, content_x])
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[title_in, content_in], outputs=x)
    print(model.summary())
    return model


def model_bilstm(vocab_size, n_classes, emb_mat):
    title_in = Input(shape=(conf.title_seq_len,), dtype='int32')
    title_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(title_in)
    title_x = Dropout(0.5)(title_x)
    title_x = Bidirectional(LSTM(conf.emb_hidden_size))(title_x)

    content_in = Input(shape=(conf.content_seq_len,), dtype='int32')
    content_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(content_in)
    content_x = Dropout(0.5)(content_x)
    content_x = Bidirectional(LSTM(conf.emb_hidden_size))(content_x)

    x = add([title_x, content_x])
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[title_in, content_in], outputs=x)
    print(model.summary())
    return model


def model_rcnn(vocab_size, n_classes, emb_mat):
    title_in = Input(shape=(conf.title_seq_len,), dtype='int32')
    title_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(title_in)
    title_x = Conv1D(64, 3, padding='same')(title_x)
    title_x = BatchNormalization()(title_x)
    title_x = Activation('relu')(title_x)
    title_x = MaxPooling1D()(title_x)
    title_x = Bidirectional(LSTM(conf.emb_hidden_size, ))(title_x)

    content_in = Input(shape=(conf.content_seq_len,), dtype='int32')
    content_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(content_in)
    content_x = Conv1D(64, 3, padding='same')(content_x)
    content_x = BatchNormalization()(content_x)
    content_x = Activation('relu')(content_x)
    content_x = MaxPooling1D()(content_x)
    content_x = Bidirectional(LSTM(conf.emb_hidden_size))(content_x)

    x = add([title_x, content_x])
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[title_in, content_in], outputs=x)
    print(model.summary())
    return model


def model_cnn(vocab_size, n_classes, emb_mat):
    filters = 64
    title_in = Input(shape=(conf.title_seq_len,), dtype='int32')
    title_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(title_in)
    title_x = Conv1D(filters, 3, padding='same')(title_x)
    title_x = BatchNormalization()(title_x)
    title_x = Activation('relu')(title_x)
    title_x = GlobalMaxPool1D()(title_x)

    content_in = Input(shape=(conf.content_seq_len,), dtype='int32')
    content_x = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(content_in)
    content_x = Conv1D(filters, 3, padding='same')(content_x)
    content_x = BatchNormalization()(content_x)
    content_x = Activation('relu')(content_x)
    content_x = GlobalMaxPool1D()(content_x)

    x = add([title_x, content_x])
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[title_in, content_in], outputs=x)
    print(model.summary())
    return model


def model_rcnn_res(vocab_size, n_classes, emb_mat):
    filters = 150
    title_in = Input(shape=(conf.title_seq_len,), dtype='int32')
    title_em = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(title_in)
    title_x = Conv1D(filters, 3, padding='same')(title_em)
    title_x = BatchNormalization()(title_x)
    title_x = Activation('relu')(title_x)
    title_x = Conv1D(conf.emb_hidden_size, 3, padding='same')(title_x)
    title_x = BatchNormalization()(title_x)
    title_x = add([title_em, title_x])
    title_x = Activation('relu')(title_x)

    title_x = Bidirectional(LSTM(conf.emb_hidden_size))(title_x)

    content_in = Input(shape=(conf.content_seq_len,), dtype='int32')
    content_em = Embedding(vocab_size, conf.emb_hidden_size, weights=[emb_mat], trainable=False)(content_in)

    content_x = Conv1D(filters, 3, padding='same')(content_em)
    content_x = BatchNormalization()(content_x)
    content_x = Activation('relu')(content_x)
    content_x = Conv1D(conf.emb_hidden_size, 3, padding='same')(content_x)
    content_x = BatchNormalization()(content_x)
    content_x = add([content_em, content_x])
    content_x = Activation('relu')(content_x)

    content_x = Bidirectional(LSTM(conf.emb_hidden_size))(content_x)

    x = add([title_x, content_x])
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[title_in, content_in], outputs=x)
    print(model.summary())
    return model


def model_cnn_res(vocab_size, n_classes, emb_mat):
    filters = 128
    title_in = Input(shape=(conf.title_seq_len,), dtype='int32')
    title_em = Embedding(vocab_size, conf.emb_size, weights=[emb_mat], trainable=False)(title_in)

    title_x1 = Conv1D(filters, 3, padding='same')(title_em)
    title_x1 = BatchNormalization()(title_x1)
    title_x1 = Activation('relu')(title_x1)

    title_x = Conv1D(conf.emb_hidden_size, 3, padding='same')(title_x1)
    title_x = BatchNormalization()(title_x)
    title_x = add([title_em, title_x])
    title_x = Activation('relu')(title_x)

    title_x = GlobalMaxPool1D()(title_x)

    # content
    content_in = Input(shape=(conf.content_seq_len,), dtype='int32')
    content_em = Embedding(vocab_size, conf.emb_size, weights=[emb_mat], trainable=False)(content_in)

    content_x1 = Conv1D(filters, 3, padding='same')(content_em)
    content_x1 = BatchNormalization()(content_x1)
    content_x1 = Activation('relu')(content_x1)

    content_x = Conv1D(conf.emb_hidden_size, 3, padding='same')(content_x1)
    content_x = BatchNormalization()(content_x)
    content_x = add([content_em, content_x])
    content_x = Activation('relu')(content_x)

    content_x = GlobalMaxPool1D()(content_x)

    x = add([title_x, content_x])
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=[title_in, content_in], outputs=x)
    print(model.summary())
    return model
