import os
import torch
import time
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import get_time_dif
from tqdm import tqdm
from torch.utils.data import DataLoader
import pdb
from utils.utils import save_model, load_model
import numpy as np
from processor.data_loader_cls import collate_fn_data
from utils.metric import evaluate_cls


class Framework(object):
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model.to(config.device)
        self.logger = logger

    def load_model(self, model_path):
        self.model = load_model(self.model, model_path)

    def set_parameter_learning_rate(self):
        # optimization setting for bert and other parameters
        if not hasattr(self.model, 'bert'):
            optimizer_grouped_parameters = [{
                'params': [p for n, p in list(self.model.named_parameters())],
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate
            }]
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            bert_param_optimizer = list(self.model.bert.named_parameters())
            bert_param_optimizer_names = [n for n, p in bert_param_optimizer]
            other_param_optimizer = [(n, p) for n, p in list(self.model.named_parameters())
                                     if not any(bn in n for bn in bert_param_optimizer_names)]
            other_param_optimizer_names = [n for n, p in other_param_optimizer]
            optimizer_grouped_parameters = [{
                'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                self.config.weight_decay,
                'lr':
                self.config.learning_rate
            }, {
                'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.config.learning_rate
            }, {
                'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay':
                self.config.weight_decay,
                'lr':
                self.config.other_learning_rate
            }, {
                'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.config.other_learning_rate
            }]

        return optimizer_grouped_parameters

    def train(self, train_dataset, dev_dataset):
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.config.batch_size,
                                      shuffle=True,
                                      collate_fn=collate_fn_data)

        optimizer_grouped_parameters = self.set_parameter_learning_rate()
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        train_steps = len(train_dataloader) * self.config.max_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=train_steps * self.config.warmup,
                                                    num_training_steps=train_steps)

        start_time = time.time()
        best_f1, best_epoch, best_loss, best_p, best_r = 0, 0, 0, 0, 0
        x, y_loss, y_p, y_r, y_f1 = [], [], [], [], []
        for epoch_index in range(self.config.max_epoch):
            self.logger.info('epoch {}/{}'.format(epoch_index, self.config.max_epoch))
            self.model.train()
            bar_train = tqdm(train_dataloader)
            for i, batch in enumerate(bar_train):
                guids, tokens_ids, segs_ids, att_mask, candidates_span, labels_span, token_start_idxs_ori2sub = batch
                tokens_ids = torch.tensor(tokens_ids, dtype=torch.long).to(self.config.device)
                segs_ids = torch.tensor(segs_ids, dtype=torch.long).to(self.config.device)
                att_mask = torch.tensor(att_mask, dtype=torch.long).to(self.config.device)

                optimizer.zero_grad()
                loss, _ = self.model(tokens_ids, att_mask, segs_ids, candidates_span, labels_span)
                loss.backward()
                train_loss = loss.item()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                optimizer.step()
                scheduler.step()
                bar_train.set_description('loss: {:.4f}'.format(train_loss))
                torch.cuda.empty_cache()

                if self.config.logging_steps != -1 and (i + 1) % self.config.logging_steps == 0:
                    # validate
                    self.logger.info('{}/{} epoch {}/{} step train finish, time:{}'.format(
                        epoch_index, self.config.max_epoch, i, len(bar_train), get_time_dif(start_time)))
                    loss, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = self.evaluate(
                        dev_dataset, type_num_odict=self.config.label2id, show_type_result=False)
                    dev_loss, dev_p, dev_r, dev_f1 = loss, micro_p, micro_r, micro_f1
                    x, y_loss, y_p, y_r, y_f1 = x + [epoch_index], y_loss + [dev_loss
                                                                             ], y_p + [dev_p], y_r + [dev_r], y_f1 + [dev_f1]
                    improve = ''
                    if dev_f1 >= best_f1:
                        improve = '*'  # if improved
                        best_epoch, best_loss, best_p, best_r, best_f1 = epoch_index, dev_loss, dev_p, dev_r, dev_f1
                        save_model(
                            self.model,
                            os.path.join(self.config.save_model_dir, self.config.data_type, self.config.prefix,
                                         self.config.saved_model_name))
                        self.logger.info(
                            'epoch_index: {} | dev-loss: {:.4f} | dev_p: {:.4f} | dev_r: {:.4f} | dev-f1: {:.4f} | time: {} | {}'
                            .format(epoch_index, dev_loss, dev_p, dev_r, dev_f1, get_time_dif(start_time), improve))

            # validate
            self.logger.info('{}/{} epoch train finish, time:{}'.format(epoch_index, self.config.max_epoch,
                                                                        get_time_dif(start_time)))

            loss, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1 = self.evaluate(dev_dataset,
                                                                                         type_num_odict=self.config.label2id,
                                                                                         show_type_result=False)
            dev_loss, dev_p, dev_r, dev_f1 = loss, micro_p, micro_r, micro_f1
            x, y_loss, y_p, y_r, y_f1 = x + [epoch_index], y_loss + [dev_loss], y_p + [dev_p], y_r + [dev_r], y_f1 + [dev_f1]
            improve = ''
            if dev_f1 >= best_f1:
                improve = '*'  # if improved
                best_epoch, best_loss, best_p, best_r, best_f1 = epoch_index, dev_loss, dev_p, dev_r, dev_f1
                save_model(
                    self.model,
                    os.path.join(self.config.save_model_dir, self.config.data_type, self.config.prefix,
                                 self.config.saved_model_name))

            self.logger.info(
                'epoch_index: {} | dev-loss: {:.4f} | dev_p: {:.4f} | dev_r: {:.4f} | dev-f1: {:.4f} | time: {} | {}'.format(
                    epoch_index, dev_loss, dev_p, dev_r, dev_f1, get_time_dif(start_time), improve))
            # early stop
            if epoch_index - best_epoch > self.config.patience:
                self.logger.info("No optimization for {} epoch, stop training...".format(self.config.patience))
                break
        self.logger.info('Final!')
        self.logger.info('best_epoch: {} | dev-loss: {:.4f} | dev-p: {:.4f} | dev-r: {:.4f} | dev-f1: {:.4f} | time:{}'.format(
            best_epoch, best_loss, best_p, best_r, best_f1, get_time_dif(start_time)))

    def fetch_label_from_probability(self, p_preds, mark):
        if mark == 'softmax':
            # p_preds: [b, st, l]
            label_preds_padded = torch.argmax(p_preds, dim=-1)
        elif mark == 'sigmoid':
            # p_preds: [b, st, l-1]
            mask_threshold = torch.max(p_preds, dim=-1)[0] < self.config.threshold_span  # small than threshold, set to True
            label_preds_padded_without_none = torch.argmax(p_preds, dim=-1) + 1  # storage the labels preds, [b, st]
            label_preds_padded = torch.masked_fill(label_preds_padded_without_none, mask_threshold, 0)
        return label_preds_padded

    def evaluate(self, dev_dataset, show_type_result=False, type_num_odict=None):
        data_loader = DataLoader(dev_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn_data)
        self.logger.info('\nevaluate the model...')
        loss = 0
        flag_once = False
        self.model.eval()
        sent_id_global = 0
        bar_dev = data_loader

        labels_truth = []
        labels_preds = []
        for batch in bar_dev:
            guids, tokens_ids, segs_ids, att_mask, candidates_span, labels_span, token_start_idxs_ori2sub = batch
            tokens_ids = torch.tensor(tokens_ids, dtype=torch.long).to(self.config.device)
            segs_ids = torch.tensor(segs_ids, dtype=torch.long).to(self.config.device)
            att_mask = torch.tensor(att_mask, dtype=torch.long).to(self.config.device)

            with torch.no_grad():
                batch_loss, p_preds = self.model(tokens_ids, att_mask, segs_ids, candidates_span, labels_span)

            loss += batch_loss.item()
            label_preds_padded = self.fetch_label_from_probability(p_preds, mark=self.config.pred_layer)
            label_preds_padded = label_preds_padded.cpu().numpy().tolist()

            labels_preds_batched_flat = []
            labels_truth_batched_flat = []
            for i in range(len(candidates_span)):
                num_valid = len(candidates_span[i])  # valid candidates without padding
                assert len(labels_span[i]) == num_valid
                labels_preds_batched_flat.extend(label_preds_padded[i][:num_valid])
                labels_truth_batched_flat.extend(labels_span[i])

            labels_preds.extend(labels_preds_batched_flat)
            labels_truth.extend(labels_truth_batched_flat)

        micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, t = evaluate_cls(preds=labels_preds,
                                                                                 labels=labels_truth,
                                                                                 report=show_type_result,
                                                                                 label2id=type_num_odict)
        self.logger.info("micro - precision: {:.4f} - recall: {:.4f} - f1: {:.4f}".format(micro_p, micro_r, micro_f1))
        self.logger.info("macro - precision: {:.4f} - recall: {:.4f} - f1: {:.4f}".format(macro_p, macro_r, macro_f1))
        if t:
            self.logger.info(t)
        return loss / len(data_loader), micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1

    def predict(self, dev_dataset):
        data_loader = DataLoader(dev_dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate_fn_data)

        self.logger.info('\n predict ...')
        loss = 0
        flag_once = False
        self.model.eval()
        sent_id_global = 0
        bar_dev = tqdm(data_loader)

        labels_preds_dataset = []
        candidates_spans_dataset = []
        for batch in bar_dev:
            guids, tokens_ids, segs_ids, att_mask, candidates_span, labels_span, token_start_idxs_ori2sub = batch
            tokens_ids = torch.tensor(tokens_ids, dtype=torch.long).to(self.config.device)
            segs_ids = torch.tensor(segs_ids, dtype=torch.long).to(self.config.device)
            att_mask = torch.tensor(att_mask, dtype=torch.long).to(self.config.device)

            with torch.no_grad():
                batch_loss, p_preds = self.model(tokens_ids,
                                                 att_mask,
                                                 segs_ids,
                                                 candidate_span=candidates_span,
                                                 label_span=None)

            label_preds_padded = self.fetch_label_from_probability(p_preds, mark=self.config.pred_layer)
            label_preds_padded = label_preds_padded.cpu().numpy().tolist()

            labels_preds_batched_flat = []
            candidates_spans_batched = []
            for i in range(len(candidates_span)):
                num_valid = len(candidates_span[i])  # valid candidates without padding
                labels_preds_batched_flat.append(label_preds_padded[i][:num_valid])  # prediction for each candidate

                candidates_spans_sent = []
                for each_candidate in candidates_span[i]:
                    if each_candidate[0] in token_start_idxs_ori2sub[i]:
                        sta = list(token_start_idxs_ori2sub[i]).index(each_candidate[0])
                    if each_candidate[1] + 1 in token_start_idxs_ori2sub[i]:
                        end = list(token_start_idxs_ori2sub[i]).index(each_candidate[1] + 1) - 1
                    candidates_spans_sent.append([sta, end])  # candidates in each sentence
                candidates_spans_batched.append(candidates_spans_sent)

            labels_preds_dataset.extend(labels_preds_batched_flat)
            candidates_spans_dataset.extend(candidates_spans_batched)
        predictions_all = []
        for sent_i in range(len(labels_preds_dataset)):
            label_preds = labels_preds_dataset[sent_i]
            candidates_spans = candidates_spans_dataset[sent_i]
            predictions_sent = []
            for span_j in range(len(candidates_spans)):
                predictions_sent.append([label_preds[span_j], candidates_spans[span_j][0], candidates_spans[span_j][1]])
            predictions_all.append(predictions_sent)
        return predictions_all
