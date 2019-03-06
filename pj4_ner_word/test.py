import torch as t

import pickle
import itertools
from torch.autograd import Variable
import numpy as np
import time
ts = time.time()


def find_length_from_labels(labels, label_to_ix):
    """ find length of unpadded features based on labels """
    end_position = len(labels) - 1
    for position, label in enumerate(labels):
        if label == label_to_ix['<pad>']:
            end_position = position
            break
    return end_position


def calc_f1_batch(target_data, decoded_data_crfs):
    num = 0
    for target, decoded_data_crf in zip(target_data, decoded_data_crfs):
        length = find_length_from_labels(target, l_map)
        num += length
        gold = target[:length]
        decoded_data_crf = decoded_data_crf[:length]
        eval_instance(decoded_data_crf, gold)
    return num


def iobes_to_spans(sequence, lut, strict_iob2=False):
    """ convert to iobes to span """
    iobtype = 2 if strict_iob2 else 1
    chunks = []
    current = None

    for i, y in enumerate(sequence):
        label = lut[y]
        if label.startswith('B-'):
            if current is not None:
                chunks.append('@'.join(current))
            current = [label.replace('B-', ''), '%d' % i]

        elif label.startswith('S-'):
            if current is not None:
                chunks.append('@'.join(current))
                current = None
            base = label.replace('S-', '')
            chunks.append('@'.join([base, '%d' % i]))

        elif label.startswith('I-'):
            if current is not None:
                base = label.replace('I-', '')
                if base == current[0]:
                    current.append('%d' % i)
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
            else:
                current = [label.replace('I-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')

        elif label.startswith('E-'):
            if current is not None:
                base = label.replace('E-', '')
                if base == current[0]:
                    current.append('%d' % i)
                    chunks.append('@'.join(current))
                    current = None
                else:
                    chunks.append('@'.join(current))
                    if iobtype == 2:
                        print('Warning')
                    current = [base, '%d' % i]
                    chunks.append('@'.join(current))
                    current = None
            else:
                current = [label.replace('E-', ''), '%d' % i]
                if iobtype == 2:
                    print('Warning')
                chunks.append('@'.join(current))
                current = None
        else:
            if current is not None:
                chunks.append('@'.join(current))
            current = None

    if current is not None:
        chunks.append('@'.join(current))

    return set(chunks)


def eval_instance(best_path, gold):
    total_labels = len(best_path)
    correct_labels = np.sum(np.equal(best_path, gold))

    gold_chunks = iobes_to_spans(gold, r_l_map)
    gold_count = len(gold_chunks)

    guess_chunks = iobes_to_spans(best_path, r_l_map)
    guess_count = len(guess_chunks)

    overlap_chunks = gold_chunks & guess_chunks
    overlap_count = len(overlap_chunks)

    return correct_labels, total_labels, gold_count, guess_count, overlap_count


if __name__ == '__main__':

    with open('data/datas.pkl', 'rb') as f:
        datas = pickle.load(f)
        dev_dataset_loader = datas['dev_dataset_loader']
        l_map = datas['CRF_l_map']
        del datas
    r_l_map = {v: k for k, v in l_map.items()}

    path = 'data/model.pth'
    model = t.load(path)
    model.eval()
    tot_length = 0

    for w_f, tg, mask_v, len_v, cnn_features in itertools.chain.from_iterable(dev_dataset_loader):
        with t.no_grad():
            mlen = len_v.max(0)[0].squeeze()
            w_f = Variable(w_f[:, 0:mlen[1]].transpose(0, 1)).cuda()
            mask_v = Variable(mask_v[:, 0:mlen[1]].transpose(0, 1)).cuda()
            cnn_features = Variable(cnn_features[:, 0:mlen[1], 0:mlen[2]].transpose(0, 1)).cuda().contiguous()

        word_representations = model.word_rep(w_f, cnn_features)
        tg = tg.numpy() % len(l_map)
        decoded_crf = model.crf.decode(word_representations, mask_v).data
        decoded_crf = decoded_crf.cpu().transpose(0, 1).numpy()
        tot_length += calc_f1_batch(tg, decoded_crf)

    ti = time.time() - ts
    print(tot_length)
    print(ti / tot_length * 1000)