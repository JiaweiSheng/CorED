# coding:utf-8
import argparse
import os
import torch, random
import numpy as np
import time
from datetime import timedelta
import json
import pdb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_text_data(f_name):
    with open(f_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def write_text_data(lines, f_name):
    with open(f_name, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def read_jsonl(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines_new = []
    for line in lines:
        line_new = json.loads(line)
        lines_new.append(line_new)
    return lines_new


def write_jsonl(fn, lines):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in lines:
            line_str = json.dumps(line, ensure_ascii=False)
            f.write(line_str + '\n')
    return True


def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, model_path, strict=False):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    return model


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_num_line(f):
    num_line = sum([1 for _ in f])
    f.seek(0, 0)
    return num_line


def output2pre(output, threshold):
    predict = output.cpu().numpy()
    predict[predict <= threshold] = 0
    predict[predict > threshold] = 1
    return predict.tolist()
