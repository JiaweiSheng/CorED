from collections import Counter
from os import O_APPEND
import numpy as np
import pdb
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label, self.markup)
            pre_entities = get_entities(pre_path, self.id2label, self.markup)  # [['PER', 0, 1], ['LOC', 3, 3]]
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def get_entity_span(s_p, e_p, max_span_len, threshold_sta, threshold_end, mode, id2label=None):
    assert mode in ['sigmoid', 'softmax']
    if mode == 'sigmoid':
        span_triples = get_entity_span_sigmoid(s_p, e_p, max_span_len, threshold_sta, threshold_end, id2label)
    else:
        span_triples = get_entity_span_softmax(s_p, e_p, max_span_len, threshold_sta, threshold_end, id2label)
    return span_triples


def get_entity_span_sigmoid(s_p, e_p, max_span_len, threshold_sta, threshold_end, id2label=None):
    """Gets entities from sequence.
    note: SE 
    Args:
        s_p (np.array): [max_sent_len, label]
        e_p (np.array): [max_sent_len, label]
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    """

    s_labels = []
    s_max = np.max(s_p, axis=-1)
    for i, max_score in enumerate(s_max):
        if max_score > threshold_sta:
            s_type = np.argmax(s_p[i]) + 1  # 0 for non-type
            s_labels.append(s_type)
        else:
            s_labels.append(0)

    e_labels = []
    e_max = np.max(e_p, axis=-1)
    for i, max_score in enumerate(e_max):
        if max_score > threshold_end:
            e_type = np.argmax(e_p[i]) + 1  # 0 for non-type
            e_labels.append(e_type)
        else:
            e_labels.append(0)

    span_triples = []
    for span_sta, s_label in enumerate(s_labels):
        if s_label != 0:
            for offset, e_label in enumerate(e_labels[span_sta:]):
                if offset + 1 > max_span_len:
                    break
                if e_label == s_label:
                    span_type = s_label
                    span_end = span_sta + offset
                    span_triples.append((span_type, span_sta, span_end))
                    break
    return span_triples


def get_entity_span_softmax(s_p, e_p, max_span_len, threshold_sta, threshold_end, id2label=None):
    """Gets entities from sequence.
    note: SE 
    Args:
        s_p (np.array): [max_sent_len, label]
        e_p (np.array): [max_sent_len, label]
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    """

    s_labels = np.argmax(s_p, axis=-1)
    e_labels = np.argmax(e_p, axis=-1)

    span_triples = []
    for span_sta, s_label in enumerate(s_labels):
        if s_label != 0:
            for offset, e_label in enumerate(e_labels[span_sta:]):
                if offset + 1 > max_span_len:
                    break
                if e_label == s_label:
                    span_type = s_label
                    span_end = span_sta + offset
                    span_triples.append((span_type, span_sta, span_end))
                    break
    return span_triples


def gen_pre_events(s_p, e_p):
    '''
        fetch all the results, given the predicted sequence
        input:(max_len, label-1)
        output:[(type, begin, end)]
    '''

    predict_events = []
    for word_index, word_l in enumerate(zip(s_p, e_p)):
        word_s_p = word_l[0]
        for word_dim, flag in enumerate(word_s_p):
            if flag == 1:
                type = word_dim + 1
                begin = word_index
                for i, word_e_p in enumerate(e_p[word_index:]):
                    if word_e_p[word_dim] == 1:
                        end = i + word_index
                        predict_events.append((type, begin, end))
                        break
                    if i > 4:
                        break
    return predict_events


def evaluate_cls(preds, labels, label2id: dict, report=False):
    num_labels = len(label2id)
    pos_labels = list(range(1, num_labels))
    labels = np.array(labels)
    preds = np.array(preds)
    micro_p = precision_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    micro_r = recall_score(labels, preds, labels=pos_labels, average='micro') * 100.0
    micro_f1 = f1_score(labels, preds, labels=pos_labels, average='micro') * 100.0

    macro_p = precision_score(labels, preds, labels=pos_labels, average='macro') * 100.0
    macro_r = recall_score(labels, preds, labels=pos_labels, average='macro') * 100.0
    macro_f1 = f1_score(labels, preds, labels=pos_labels, average='macro') * 100.0

    t = None
    if report and label2id:
        label_names = list(label2id.keys())
        t = classification_report(y_true=labels, y_pred=preds, target_names=label_names)
    return micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, t
