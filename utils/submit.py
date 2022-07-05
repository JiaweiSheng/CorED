from utils.utils import read_jsonl, write_jsonl


def obtain_submit_format(fn_original_test, fn_output, predictions, label2id=None):
    lines = read_jsonl(fn_original_test)
    res_all = []
    cur_index_prediction = 0
    for doc in lines:
        res = {}
        res['id'] = doc['id']
        res['predictions'] = []
        num_sentences = len(doc['content'])
        cur_predictions = predictions[cur_index_prediction:cur_index_prediction + num_sentences]
        # prediction: (type, sta, end)
        pre_dict = {}  # (sent_id, sta, end+1) = type_id
        for sent_id, prediction in enumerate(cur_predictions):
            for type_id, sta, end in prediction:
                key = (sent_id, sta, end + 1)
                if isinstance(type_id, str) and label2id is not None:
                    type_id = label2id[type_id]
                pre_dict[key] = int(type_id)
        for mention in doc['candidates']:
            mention_id = mention['id']
            mention_key = (mention['sent_id'], mention['offset'][0], mention['offset'][1])
            if mention_key in pre_dict:
                mention_type = pre_dict[mention_key]
                res['predictions'].append({'id': mention_id, 'type_id': mention_type})
            else:
                res['predictions'].append({'id': mention_id, 'type_id': 0})
        res_all.append(res)
        cur_index_prediction += num_sentences

    write_jsonl(fn_output, res_all)
