# coding:utf-8
import argparse
import os
import logging


def str2bool(str):
    return True if str.lower() == 'true' else False


def params(root_path):
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument

    add_arg("--train", default=True, help="Whether train the model", type=str2bool)
    add_arg("--model_name_or_path",
            default="plm/bert-base-uncased",
            type=str,
            help="Path to pre-trained model or shortcut name selected in the list: ")
    add_arg("--cuda_id", default=0, type=float)
    add_arg("--gradient_accumulation_steps", default=1, type=int)

    # Hyperparameter
    add_arg("--batch_size", default=50, help="batch size", type=int)  #
    add_arg("--max_epoch", default=10, help="maximum training epochs", type=int)
    add_arg("--max_len", default=128, help="max len of the sentence", type=int)  #
    add_arg("--max_candidate_num", default=128, help="max len of training candidates", type=int)  #
    add_arg("--label_num", default=169, help="num of the label", type=int)
    add_arg("--label_emb", default=768, help="dim of the label embedding", type=int)
    add_arg("--label_init", default=168, help="dim of input features of gcn", type=int)
    add_arg("--gcn_n_layers", default=1, help="num of gcn layer", type=int)
    add_arg("--bert_dropout_rate", default=0.1, help="dropout of decoder", type=float)
    add_arg("--decoder_dropout_rate", default=0.5, help="dropout of decoder", type=float)
    add_arg("--dropout_rate_att", default=0.1, help="dropout of decoder", type=float)

    add_arg("--clip", default=5, help="Gradient clip", type=float)
    add_arg("--learning_rate", default=1e-5, help="Initial learning rate for bert", type=float)
    add_arg("--other_learning_rate", default=5e-4, help="Initial learning rate for non-bert", type=float)

    add_arg("--weight_decay", default=0.0, help="Learning rate decay", type=float)
    add_arg("--warmup", default=0.1, help="Warmup rate", type=float)
    add_arg("--patience", default=10, help="num of epoch for early stop", type=int)
    add_arg("--steps_check", default=50, help="steps per checkpoint", type=int)
    add_arg("--threshold", default=0.5, help="threshold for classification", type=float)
    add_arg("--logging_steps", type=int, default=-1, help="Log every X updates steps.")
    add_arg("--tune_emb", default=True, help="Whether train the model", type=str2bool)

    # path
    add_arg("--data_proc_dir", default="dataset/data_processed", help="file for processed data", type=str)
    add_arg("--saved_model_name", default="pytorch.model", help="trained model named according to the params", type=str)
    add_arg("--f_train", default=os.path.join(root_path, "dataset/maven/maven_processed/train.json"), type=str)
    add_arg("--f_valid", default=os.path.join(root_path, "dataset/maven/maven_processed/valid.json"), type=str)
    add_arg("--f_test", default=os.path.join(root_path, "dataset/maven/maven_processed/test.json"), type=str)

    add_arg("--f_id2label", default=os.path.join(root_path, "dataset/maven/event_type.json"), help="", type=str)
    add_arg("--f_type_embedding",
            default=os.path.join(root_path, 'dataset/maven/label_graphs/type_literal_embeddings.npy'),
            type=str)

    add_arg("--save_model_dir", default="output", help="Path to save model", type=str)
    add_arg("--prefix", default="exp_model", help="", type=str)

    add_arg("--adjacent_threshold", default=0.3, type=float)
    add_arg("--max_span_len", default=10, type=float)

    add_arg("--encoder", default='bert', type=str)
    add_arg("--seed", default=42, type=int)

    add_arg("--gcn_act", default='LeakyRelu', type=str)
    add_arg("--gcn_layer_num", default=1, type=int)

    add_arg("--do_train", default=False, type=str2bool)
    add_arg("--do_valid", default=False, type=str2bool)
    add_arg("--do_test", default=False, type=str2bool)

    add_arg("--f_origin_test", default='dataset/maven/maven/test.jsonl', help="", type=str)
    add_arg("--f_result", default='results.jsonl', help="predicted results", type=str)
    add_arg("--variant_fusion", default='default', type=str)
    add_arg("--data_type", default='maven', type=str)

    add_arg("--graph_metric_type", default='weighted_cosine', type=str)
    add_arg("--graph_learn_epsilon_label2label", default=0.1, type=float)
    add_arg("--leakyrelu_alpha", default=0.2, type=float)
    add_arg("--nheads", default=8, type=int)
    add_arg("--nlayer", default=1, type=int)
    add_arg("--graph_learn_num_pers", default=16, type=int)

    add_arg("--threshold_span", default=0.7, type=float)
    add_arg("--pow_1", default=1, type=float)
    add_arg("--pow_2", default=2, type=float)

    add_arg("--w1", default=0, type=float)
    add_arg("--w2", default=1, type=float)
    add_arg("--agg", default='maxp', type=str)
    add_arg("--pred_layer", default='sigmoid', type=str)
    add_arg("--show_type_result", default=False, type=str2bool)
    add_arg("--num_heads", default=4, type=int)
    add_arg("--num_t2t_layer", default=1, type=int)

    args = parser.parse_args()
    return args


def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%m-%d %H:%M:%S')
    fh = logging.FileHandler('./logs/{}/{}.log'.format(args.data_type, args.prefix), mode='w+')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        logger.info(k + ': ' + str(v))
    logger.info("----------------------------")
    return logger
