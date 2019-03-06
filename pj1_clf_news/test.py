import keras
import pj1_clf_news.config as conf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from pj1_clf_news.dataset import Dataset


def test(model_name):
    # 加载数据
    test_dataset = Dataset(conf.dev_data)
    # 加载model
    model = keras.models.load_model(conf.model_path.format(model_name))
    preds = model.predict_generator(test_dataset.gen_batch_data(), steps=test_dataset.steps_per_epoch)
    preds = np.argmax(preds, axis=1)
    labels = test_dataset.tags
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds))
    print(accuracy_score(labels, preds))


if __name__ == '__main__':
    test('cnn_res')



