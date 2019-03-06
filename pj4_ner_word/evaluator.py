from __future__ import division
import numpy as np
import itertools
from torch.autograd import Variable
import torch


class Evaluator:
    """ evaluation class for ner task """

    def __init__(self, l_map):
        self.l_map = l_map
        self.r_l_map = {v: k for k, v in l_map.items()}

    def reset(self):
        self.correct_labels_crf = 0
        self.total_labels_crf = 0
        self.gold_count_crf = 0
        self.guess_count_crf = 0
        self.overlap_count_crf = 0

    def calc_f1_batch(self, target_data, decoded_data_crfs):
        """ update statics for f1 score
        args:
            decoded_data (batch_size, seq_len): prediction sequence
            target_data (batch_size, seq_len): ground-truth """
        for target, decoded_data_crf in zip(target_data, decoded_data_crfs):
            length = find_length_from_labels(target, self.l_map)
            gold = target[:length]
            decoded_data_crf = decoded_data_crf[:length]

            correct_labels_i, total_labels_i, gold_count_i, guess_count_i, overlap_count_i = self.eval_instance(
                decoded_data_crf, gold)
            self.correct_labels_crf += correct_labels_i
            self.total_labels_crf += total_labels_i
            self.gold_count_crf += gold_count_i
            self.guess_count_crf += guess_count_i
            self.overlap_count_crf += overlap_count_i

    def f1_score(self):
        """ calculate f1 score batgsed on statics """
        if self.guess_count_crf == 0:
            f_crf, precision_crf, recall_crf, accuracy_crf = 0.0, 0.0, 0.0, 0.0
        else:
            precision_crf = self.overlap_count_crf / float(self.guess_count_crf)
            recall_crf = self.overlap_count_crf / float(self.gold_count_crf)
            if precision_crf == 0.0 or recall_crf == 0.0:
                f_crf, precision_crf, recall_crf, accuracy_crf = 0.0, 0.0, 0.0, 0.0
            else:
                f_crf = 2 * (precision_crf * recall_crf) / (precision_crf + recall_crf)
                accuracy_crf = float(self.correct_labels_crf) / self.total_labels_crf

        return f_crf, precision_crf, recall_crf, accuracy_crf

    def eval_instance(self, best_path, gold):
        """ update statics for one instance
        args:
            best_path (seq_len): predicted
            gold (seq_len): ground-truth """
        total_labels = len(best_path)
        correct_labels = np.sum(np.equal(best_path, gold))

        gold_chunks = iobes_to_spans(gold, self.r_l_map)
        gold_count = len(gold_chunks)

        guess_chunks = iobes_to_spans(best_path, self.r_l_map)
        guess_count = len(guess_chunks)

        overlap_chunks = gold_chunks & guess_chunks
        overlap_count = len(overlap_chunks)

        return correct_labels, total_labels, gold_count, guess_count, overlap_count

    def calc_score(self, ner_model, dataset_loader):
        ner_model.eval()
        self.reset()

        for w_f, tg, mask_v, len_v, cnn_features in itertools.chain.from_iterable(
                dataset_loader):
            with torch.no_grad():
                mlen = len_v.max(0)[0].squeeze()
                w_f = Variable(w_f[:, 0:mlen[1]].transpose(0, 1)).cuda()
                mask_v = Variable(mask_v[:, 0:mlen[1]].transpose(0, 1)).cuda()
                cnn_features = Variable(cnn_features[:, 0:mlen[1], 0:mlen[2]].transpose(0, 1)).cuda().contiguous()

            word_representations = ner_model.word_rep(w_f, cnn_features)
            tg = tg.numpy() % len(self.l_map)
            decoded_crf = ner_model.crf.decode(word_representations, mask_v).data
            decoded_crf = decoded_crf.cpu().transpose(0, 1).numpy()
            self.calc_f1_batch(tg, decoded_crf)
        ner_model.train()
        return self.f1_score()


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


def find_length_from_labels(labels, label_to_ix):
    """ find length of unpadded features based on labels """
    end_position = len(labels) - 1
    for position, label in enumerate(labels):
        if label == label_to_ix['<pad>']:
            end_position = position
            break
    return end_position
