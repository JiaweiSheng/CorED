import argparse
from utils.params import params, get_logger
from utils.utils import seed_everything
from utils.submit import obtain_submit_format
import torch
import os
from transformers import BertTokenizer, BertConfig
from processor.data_loader_cls import get_id2label_label2id
from processor.data_loader_cls import Data
from models.model import Model
from utils.framework_cls import Framework

SAMPLE_SUBSET_NUM = -1


def main(superconfig=None):
    root_path = './'
    config = params(root_path=root_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    if superconfig:
        config = vars(config)
        for k in superconfig:
            config[k] = superconfig[k]
        config = argparse.Namespace(**config)

    seed_everything(config.seed)
    if not os.path.exists(os.path.join(config.save_model_dir, config.data_type)):
        os.makedirs(os.path.join(config.save_model_dir, config.data_type))
    if not os.path.exists(os.path.join(config.save_model_dir, config.data_type, config.prefix)):
        os.makedirs(os.path.join(config.save_model_dir, config.data_type, config.prefix))

    logger = get_logger(config)

    id2label, label2id = get_id2label_label2id(config.f_id2label)
    config.id2label = id2label
    config.label2id = label2id
    config.label_num = len(id2label.keys())

    tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path)
    bert_config = BertConfig.from_pretrained(config.model_name_or_path)
    config.hidden_size = bert_config.hidden_size
    config.vocab_size = bert_config.vocab_size

    model = Model(config=config)
    framework = Framework(config, model, logger)

    if config.do_train:
        train_dataset = Data(data_type='train',
                             f_name=config.f_train,
                             tokenizer=tokenizer,
                             max_text_length=config.max_len,
                             label2id=label2id,
                             max_candidate_num=config.max_candidate_num,
                             sample_subset_num=SAMPLE_SUBSET_NUM)
        dev_dataset = Data(data_type='dev',
                           f_name=config.f_valid,
                           tokenizer=tokenizer,
                           max_text_length=config.max_len,
                           label2id=label2id,
                           max_candidate_num=config.max_candidate_num,
                           sample_subset_num=SAMPLE_SUBSET_NUM)
        framework.train(train_dataset, dev_dataset)

    if config.do_valid:
        dev_dataset = Data(data_type='dev',
                           f_name=config.f_valid,
                           tokenizer=tokenizer,
                           max_text_length=config.max_len,
                           label2id=label2id,
                           max_candidate_num=config.max_candidate_num,
                           sample_subset_num=SAMPLE_SUBSET_NUM)
        framework.load_model(os.path.join(config.save_model_dir, config.data_type, config.prefix, config.saved_model_name))
        framework.evaluate(dev_dataset, show_type_result=config.show_type_result, type_num_odict=label2id)

    if config.do_test:
        test_dataset = Data(data_type='test',
                            f_name=config.f_test,
                            tokenizer=tokenizer,
                            max_text_length=config.max_len,
                            label2id=label2id,
                            max_candidate_num=config.max_candidate_num,
                            sample_subset_num=SAMPLE_SUBSET_NUM)
        framework.load_model(os.path.join(config.save_model_dir, config.data_type, config.prefix, config.saved_model_name))
        predictions_all = framework.predict(test_dataset)
        output_path = os.path.join(config.save_model_dir, config.data_type, config.prefix, 'results.jsonl')
        id2type, type2id = get_id2label_label2id(config.f_id2label)
        obtain_submit_format(config.f_origin_test, output_path, predictions_all, label2id=type2id)


if __name__ == '__main__':
    main()
