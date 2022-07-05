# -*- coding: utf-8 -*-
from torch import nn
import torch
from transformers.activations import ACT2FN
import math
import numpy as np

np.set_printoptions(threshold=np.inf)

BertLayerNorm = torch.nn.LayerNorm


class MaskedTransformerConfig(object):
    def __init__(self,
                 max_len,
                 hidden_size,
                 label_emb,
                 num_heads,
                 num_hidden_layers=1,
                 num_attention_heads=1,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 layer_norm_eps=1e-12,
                 partial_att=False):
        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.label_emb = label_emb
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.num_heads = num_heads
        self.partial_att = partial_att


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # LN
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor=None):
        # dense -> dp -> res -> LN
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if input_tensor is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MaskedAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout_att = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = BertSelfOutput(config)
        self.linear_q = nn.Linear(config.hidden_size + config.label_emb, config.hidden_size + config.label_emb)
        self.linear_k = nn.Linear(config.hidden_size + config.label_emb, config.hidden_size + config.label_emb)
        self.linear_v = nn.Linear(config.hidden_size + config.label_emb, config.hidden_size + config.label_emb)

    def forward(
        self,
        hidden_states_truth,
        hidden_states_unknown,
        attention_mask=None,
        output_attentions=False,
    ):
        '''
        hidden_states_truth: [b, m, h+e]
        hidden_states_unknown: [b, m, h+e]
        '''
        n_batch, max_len, n_emb = hidden_states_truth.size()

        hidden_states_unknown_q = self.linear_q(hidden_states_unknown)  # [b, m, h+e]
        hidden_states_unknown_k = self.linear_k(hidden_states_unknown)  # [b, m, h+e]
        hidden_states_unknown_v = self.linear_v(hidden_states_unknown)  # [b, m, h+e]

        hidden_states_truth_k = self.linear_k(hidden_states_truth)  # [b, m, h+e]
        hidden_states_truth_v = self.linear_v(hidden_states_truth)  # [b, m, h+e]

        # attention matrix
        eye = torch.eye(max_len, max_len).to(hidden_states_truth.device)
        attention_scores_1 = torch.matmul(hidden_states_unknown_q,
                                          hidden_states_truth_k.transpose(-1, -2))  # [b, m, h+e] * [b, h+e, m] = [b, m, m]
        attention_scores_2 = torch.sum(hidden_states_unknown_q * hidden_states_unknown_k, dim=-1,
                                       keepdim=True).expand(n_batch, max_len, max_len)  # [b, m, m]
        attention_scores = attention_scores_1 * (1 - eye) + attention_scores_2 * eye  #  [b, m, m]
        attention_scores = attention_scores / math.sqrt(n_emb)  # final attention scores

        if attention_mask is not None:
            attention_mask_matrix = ((1 - attention_mask).float() * -1e9).unsqueeze(-1).expand(n_batch, max_len,
                                                                                               max_len)  # for padding
            attention_scores = attention_scores + attention_mask_matrix
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [b, m, m]
        attention_probs = self.dropout_att(attention_probs)  # [b, m, m]

        # for v matrix
        eye_exp = eye.unsqueeze(-1).expand(max_len, max_len, n_emb)  # [m, m] -> [m, m, h]
        hidden_states_truth_exp = hidden_states_truth_v.unsqueeze(1).expand(n_batch, max_len, max_len,
                                                                            n_emb)  # [b, m, h] -> [b, m, m, h]
        hidden_states_unknown_exp = hidden_states_unknown_v.unsqueeze(1).expand(n_batch, max_len, max_len,
                                                                                n_emb)  # [b, m, h] -> [b, m, m, h]
        hidden_states_exp = hidden_states_truth_exp * (1 - eye_exp) + hidden_states_unknown_exp * eye_exp  # [b, m, m, h]

        # fuse atteniton scores and v matrix
        attention_probs_exp = attention_probs.unsqueeze(-1)  # [b, m, m] -> [b, m, m, 1]
        context_layer = torch.matmul(hidden_states_exp.transpose(-1, -2), attention_probs_exp).squeeze(
            -1)  # [b, m, h+e, m] * [b, m, m, 1] -> [b, m, h+e, 1] -> [b, m, h+e]

        outputs = self.output(context_layer)  # [b, m, h]
        outputs = (outputs, attention_probs) if output_attentions else (outputs, )
        return outputs


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dropout_att = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = BertSelfOutput(config)
        self.num_heads = config.num_heads
        self.dim_in = self.dim_k = self.dim_v = config.hidden_size
        self.partial_att = config.partial_att

        self.linear_q = nn.Linear(self.dim_in, self.dim_k)
        self.linear_k = nn.Linear(self.dim_in, self.dim_k)
        self.linear_v = nn.Linear(self.dim_in, self.dim_v)

    def forward(
        self,
        hidden_states_truth,
        hidden_states_unknown,
        attention_mask=None,
        sim_first=None,
        output_attentions=False,
    ):
        '''
        hidden_states_truth: [b, m, h+e]
        hidden_states_unknown: [b, m, h+e]
        '''
        n_batch, max_len, n_emb = hidden_states_truth.size()
        assert n_emb == self.dim_in

        num_heads = self.num_heads
        dk = self.dim_k // num_heads  # dim_k(dim_q) of each head
        dv = self.dim_v // num_heads  # dim_v of each head

        hidden_states_unknown_q = self.linear_q(hidden_states_unknown).reshape(n_batch, max_len, num_heads,
                                                                               dk).transpose(1, 2)  # [b, nh, m, dk]
        hidden_states_unknown_k = self.linear_k(hidden_states_unknown).reshape(n_batch, max_len, num_heads,
                                                                               dk).transpose(1, 2)  # [b, nh, m, dk]
        hidden_states_unknown_v = self.linear_v(hidden_states_unknown).reshape(n_batch, max_len, num_heads,
                                                                               dv).transpose(1, 2)  # [b, nh, m, dv]

        hidden_states_truth_k = self.linear_k(hidden_states_truth).reshape(n_batch, max_len, num_heads,
                                                                           dk).transpose(1, 2)  # [b, nh, m, dk]
        hidden_states_truth_v = self.linear_v(hidden_states_truth).reshape(n_batch, max_len, num_heads,
                                                                           dv).transpose(1, 2)  # [b, nh, m, dv]

        # attention matrix
        eye = torch.eye(max_len, max_len).to(hidden_states_truth.device)
        attention_scores_1 = torch.matmul(hidden_states_unknown_q, hidden_states_truth_k.transpose(
            -1, -2))  # [b, nh, m, dk].dot[b, nh, dk, m] = [b, nh, m, m]
        attention_scores_2 = torch.sum(hidden_states_unknown_q * hidden_states_unknown_k, dim=-1,
                                       keepdim=True).expand_as(attention_scores_1)  # [b, nh, m, m]

        attention_scores = attention_scores_1 * (1 - eye) + attention_scores_2 * eye  #   [b, nh, m, m]
        attention_scores = attention_scores / math.sqrt(n_emb)  # final attention scores

        if attention_mask is not None:
            attention_mask_matrix = ((1 - attention_mask).float() * -1e9).view(n_batch, 1, 1,
                                                                               max_len).expand_as(attention_scores)  # padding
            attention_scores = attention_scores + attention_mask_matrix
        attention_probs = torch.softmax(attention_scores, dim=-1)  # [b, nh, m, m]

        attention_probs = self.dropout_att(attention_probs)  # [b, nh, m, m]

        eye_exp = eye.unsqueeze(-1).expand(max_len, max_len, dv)  # [m, m] -> [m, m, dv]
        hidden_states_truth_exp = hidden_states_truth_v.unsqueeze(2).expand(n_batch, num_heads, max_len, max_len,
                                                                            dv)  # [b, nh, m, dv] -> [b, nh, m, m, dv]
        hidden_states_unknown_exp = hidden_states_unknown_v.unsqueeze(2).expand(n_batch, num_heads, max_len, max_len,
                                                                                dv)  # [b, nh, m, dv] -> [b, nh, m, m, dv]
        hidden_states_exp = hidden_states_truth_exp * (1 - eye_exp) + hidden_states_unknown_exp * eye_exp  # [b, nh, m, m, dv]

        # fuse attention and v matrix
        attention_probs_exp = attention_probs.unsqueeze(-1)  # [b, nh, m, m] -> [b, nh, m, m, 1]
        context_layer = torch.matmul(hidden_states_exp.transpose(-1, -2), attention_probs_exp).squeeze(
            -1)  # [b, nh, m, dv, m].dot[b, nh, m, m, 1] -> [b, nh, m, dv, 1] -> [b, nh, m, dv]
        context_layer = context_layer.transpose(1, 2).reshape(n_batch, max_len, self.dim_v)  # flaten multi-head

        outputs = self.output(context_layer)  # [b, m, h]
        outputs = (outputs, attention_probs) if output_attentions else (outputs, )
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # dense -> act
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # dense -> dp -> res -> LN
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MaskedTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MaskedAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states_truth,
        hidden_states_unknown,
        attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states_truth,
            hidden_states_unknown,
            attention_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + outputs
        return outputs


class MaskedSALayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)  # multi-head
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states_truth, hidden_states_unknown, sim_first=None, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states_truth=hidden_states_truth,
                                                hidden_states_unknown=hidden_states_unknown,
                                                sim_first=sim_first,
                                                attention_mask=attention_mask,
                                                output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + outputs
        return outputs
