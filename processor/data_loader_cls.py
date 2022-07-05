from pickle import NONE
import pdb
import numpy as np
import json
from utils.utils import read_jsonl, read_text_data
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, data_type, f_name, tokenizer, max_text_length, label2id, max_candidate_num, sample_subset_num=-1):
        super(Data, self).__init__()
        self.guids, self.tokens_ids, self.segs_ids, self.att_mask, self.candidates_span, self.labels_span, self.candidates_span_ori, self.token_start_idxs_ori2sub = process_dataset(
            data_type, f_name, tokenizer, max_text_length, label2id, sample_subset_num)
        self.max_candidate_num = max_candidate_num
        self.label2id = label2id
        self.data_type = data_type

    def __len__(self):
        return len(self.tokens_ids)

    def __getitem__(self, index):
        guid = self.guids[index]
        token_id = self.tokens_ids[index]
        segs_id = self.segs_ids[index]
        att_mask = self.att_mask[index]
        candidates_span = self.candidates_span[index]
        labels_span = self.labels_span[index]
        candidates_span_ori = self.candidates_span_ori[index]
        token_start_idxs_ori2sub = self.token_start_idxs_ori2sub[index]
        if self.data_type == 'train':
            # To accelerate training, we randomly select part of negative instance.
            candidates_span, labels_span = self.sample_candidates(candidates_span, labels_span, self.label2id,
                                                                  self.max_candidate_num)
        return guid, token_id, segs_id, att_mask, candidates_span, labels_span, token_start_idxs_ori2sub

    def sample_candidates(self, candidates_span, labels_span, label2id, max_candidate_num):
        if max_candidate_num < len(candidates_span):
            pos_spans = []
            pos_labels = []
            neg_spans = []
            neg_labels = []

            num_pos = 0
            for i, label in enumerate(labels_span):
                if label != label2id['O']:
                    pos_spans.append(candidates_span[i])
                    pos_labels.append(label)
                    num_pos += 1
            candidates_span_neg = candidates_span[num_pos:]
            labels_span_neg = labels_span[num_pos:]

            index = list(range(len(candidates_span_neg)))
            np.random.shuffle(index)
            index = index[:max_candidate_num - num_pos]
            neg_spans = [candidates_span_neg[i] for i in index]
            neg_labels = [labels_span_neg[i] for i in index]

            spans = pos_spans + neg_spans
            labels = pos_labels + neg_labels
        else:
            spans = candidates_span
            labels = labels_span
        return spans, labels


def collate_fn_data(data):
    guids, tokens_ids, segs_ids, att_mask, candidates_span, labels_span, token_start_idxs_ori2sub = zip(*data)
    return guids, tokens_ids, segs_ids, att_mask, candidates_span, labels_span, token_start_idxs_ori2sub


def process_dataset(data_type, f_name, tokenizer, max_text_length, label2id, sample_subset_num=-1):
    assert data_type in ['train', 'dev', 'test']
    guids, token_records, candidates_span_ori, labels_span_records = loads_data(data_type, f_name, label2id, sample_subset_num)
    tokens_ids, segs_ids, att_mask, candidate_spans_subword, candidates_span_ori, labels_span_records, token_start_idxs_ori2sub = token_to_id_with_subwords(
        tokenizer, max_text_length, token_records, candidates_span_ori, labels_span_records)
    return guids, tokens_ids, segs_ids, att_mask, candidate_spans_subword, labels_span_records, candidates_span_ori, token_start_idxs_ori2sub


def get_id2label_label2id(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        id2label = json.load(f)
    id2label = {int(k): id2label[k] for k in id2label}
    label2id = {id2label[k]: k for k in id2label}
    return id2label, label2id


def loads_data(data_type, f_name: str, type2id: dict, sample_subset_num=-1):
    """
    sample_subset_num: For debug, we can sample a subset. >0 means subset number,  -1 means all the data.
    """
    lines = read_jsonl(f_name)
    token_records = []
    candidates_span_records = []
    labels_span_records = []
    tokens = []
    guids = []
    for line in lines:
        guids.append(line['guid'])
        tokens = line['word']
        candidates_span = line['candidates_span']
        if data_type in ['train', 'dev']:
            labels_span = [type2id[item] for item in line['labels_span']]
        else:
            # If the data type is 'test', then return sequences with all element 0
            labels_span = [0 for i in range(len(line['candidates_span']))]

        token_records.append(tokens)
        candidates_span_records.append(candidates_span)
        labels_span_records.append(labels_span)
    # For debug, select a small amount of data
    if sample_subset_num != -1:
        token_records = token_records[:sample_subset_num]
        candidates_span_records = candidates_span_records[:sample_subset_num]
        labels_span_records = labels_span_records[:sample_subset_num]
    return guids, token_records, candidates_span_records, labels_span_records


def token_to_id(tokenizer, max_text_length, token_records, cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"):
    tokens_ids = []
    segs_ids = []
    att_mask = []

    # print('token to id...')
    for i in range(len(token_records)):
        tokens = token_records[i]
        special_tokens_count = 2
        if len(tokens) > max_text_length - special_tokens_count:
            tokens = tokens[:(max_text_length - special_tokens_count)]

        len_text = len(tokens)
        tokens = [token.lower() for token in tokens]
        tokens = [cls_token] + tokens + [sep_token]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        at_msk = [1] * (len_text + 2)
        seg_ids = [0] * (len_text + 2)

        pad_len = max_text_length - len_text - 2
        token_ids += [0] * pad_len
        at_msk += [0] * pad_len
        seg_ids += [0] * pad_len
        assert len(token_ids) == len(at_msk) == len(seg_ids) == max_text_length
        tokens_ids.append(token_ids)
        segs_ids.append(seg_ids)
        att_mask.append(at_msk)  # [1,1,1,1,1,0,0]

    return tokens_ids, segs_ids, att_mask


def token_to_id_with_subwords(tokenizer,
                              max_text_length,
                              token_records,
                              candidates_span_records,
                              labels_span_records,
                              cls_token="[CLS]",
                              sep_token="[SEP]",
                              mask_token="[MASK]"):
    tokens_ids = []
    segs_ids = []
    att_mask = []
    candidate_spans_sub = []
    candidate_spans_ori = []
    token_start_idxs_ori2sub = []
    labels_span_all = []

    # print('token to id...')
    for i in range(len(token_records)):
        tokens = token_records[i]
        tokens = [token.lower() for token in tokens]
        # For BERT subwords
        subwords = list(map(tokenizer.tokenize, tokens))
        subword_lengths = list(map(len, subwords))
        tokens = [item for indices in subwords for item in indices]

        special_tokens_count = 2
        if len(tokens) > max_text_length - special_tokens_count:
            tokens = tokens[:(max_text_length - special_tokens_count)]
        len_text = len(tokens)
        tokens = [cls_token] + tokens + [sep_token]

        # use cumsum to record the start index of each word
        token_start_idxs = np.cumsum([0] + subword_lengths) + 1  # this will lead to one more end position
        token_start_idxs_ori2sub.append(token_start_idxs)  # used to recover original position for decoding
        # generate candidate span corresponding to position in subword sequence.
        candidate_spans_sub_within_sent = []
        labels_span_all_within_sent = []
        candidate_spans_ori_within_sent = []
        for j, span in enumerate(candidates_span_records[i]):
            sub_sta = token_start_idxs[span[0]]
            sub_end = token_start_idxs[span[1] + 1] - 1

            if sub_sta < max_text_length - 2 and sub_end < max_text_length - 2:
                span_subwords = [sub_sta, sub_end]
                candidate_spans_sub_within_sent.append(span_subwords)
                candidate_spans_ori_within_sent.append(span)
                labels_span_all_within_sent.append(labels_span_records[i][j])
            else:
                # print(span, [sub_sta, sub_end], 'is cut within max_text_length')
                pass
        candidate_spans_sub.append(candidate_spans_sub_within_sent)
        candidate_spans_ori.append(candidate_spans_ori_within_sent)
        labels_span_all.append(labels_span_all_within_sent)

        # tokenilize with subword
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        at_msk = [1] * (len_text + 2)
        seg_ids = [0] * (len_text + 2)

        pad_len = max_text_length - len_text - 2
        token_ids += [0] * pad_len
        at_msk += [0] * pad_len
        seg_ids += [0] * pad_len

        assert len(token_ids) == len(at_msk) == len(seg_ids) == max_text_length
        tokens_ids.append(token_ids)
        segs_ids.append(seg_ids)
        att_mask.append(at_msk)  # [1,1,1,1,1,0,0]

    return tokens_ids, segs_ids, att_mask, candidate_spans_sub, candidate_spans_ori, labels_span_all, token_start_idxs_ori2sub
