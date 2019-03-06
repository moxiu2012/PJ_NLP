import keras
from pj1_clf_news.model import *
import pj1_clf_news.config as conf
from pj1_clf_news.dataset import Dataset
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from sklearn.metrics import f1_score

name_model = {'lstm': model_lstm, 'bilstm': model_bilstm, 'rcnn': model_rcnn,
              'rcnn_res': model_rcnn_res, 'cnn_res': model_cnn_res, 'cnn': model_cnn}
model_name = 'cnn'


class MyCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        # print('epoch: ===', epoch)
        pass

    def on_epoch_end(self, epoch, logs=None):
        # print('=========epoch: ', epoch, 'loss: ', logs.get('loss'), 'acc: ', logs.get('acc'))
        pass

    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % 10 == 0:
            print('batch: ', batch + 1, 'loss: ', logs.get('loss'), 'acc: ', logs.get('acc'))


class F1(Callback):
    def __init__(self, validation_generate, steps_per_epoch):
        super(Callback, self).__init__()
        self.validation_generate = validation_generate
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        y_trues = []
        y_preds = []
        for i in range(self.steps_per_epoch):
            x_val, y_val = next(self.validation_generate)
            y_pred = self.model.predict(x_val, verbose=0)
            y_trues.extend(np.argmax(y_val, axis=1).tolist())
            y_preds.extend(np.argmax(y_pred, axis=1).tolist())
        score = f1_score(np.array(y_trues), np.array(y_preds), average='micro')
        print('\n f1 - epoch:%d - score:%.6f \n' % (epoch + 1, score))


def train():
    # load data
    train_dataset = Dataset(training=True)
    dev_dataset = Dataset(training=False)

    # model
    MODEL = name_model[model_name]
    model = MODEL(train_dataset.vocab_size, conf.n_classes, train_dataset.emb_mat)

    # callback
    my_callback = MyCallback()
    f1 = F1(dev_dataset.gen_batch_data(), dev_dataset.steps_per_epoch)
    checkpointer = ModelCheckpoint('data/{}.hdf5'.format(model_name), save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

    # train
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.categorical_crossentropy, metrics=['acc'])
    model.fit_generator(train_dataset.gen_batch_data(),
                        steps_per_epoch=train_dataset.steps_per_epoch,
                        verbose=0,
                        epochs=conf.epochs, callbacks=[my_callback, checkpointer, early_stop, f1])
    keras.models.save_model(model, conf.model_path.format(model_name))


if __name__ == '__main__':
    train()
