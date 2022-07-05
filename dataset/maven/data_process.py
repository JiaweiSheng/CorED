import json
from typing import Counter


def read_json(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines_new = []
    for line in lines:
        line_new = json.loads(line)
        lines_new.append(line_new)
    return lines_new


def write_json(fn, lines):
    with open(fn, 'w', encoding='utf-8') as f:
        for line in lines:
            line_str = json.dumps(line, ensure_ascii=False)
            f.write(line_str + '\n')
    return True


def transform_data_into_bio(data_dir, mode, dealed_data_dir):
    docs = read_json(data_dir)
    examples = []
    for doc in docs:
        words = []
        labels = []
        for sent in doc['content']:
            words.append(sent['tokens'])
            labels.append(['O' for i in range(0, len(sent['tokens']))])
        if mode != 'test':
            for event in doc['events']:
                for mention in event['mention']:
                    labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "I-" + event['type']
            for mention in doc['negative_triggers']:
                labels[mention['sent_id']][mention['offset'][0]] = "O"
                for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                    labels[mention['sent_id']][i] = "O"
        for i in range(0, len(words)):
            record = {'guid': "%s-%s-%d" % (mode, doc['id'], i), 'word': words[i], 'labels': labels[i]}
            examples.append(record)

    write_json(dealed_data_dir, examples)
    return examples


def transform_data_into_span(data_dir, mode, dealed_data_dir):
    docs = read_json(data_dir)
    # a = []

    examples = []
    for doc in docs:
        words = []
        labels_sta = []
        labels_end = []
        candidates = []
        candidates_sta = []
        candidates_end = []
        candidates_span = []
        labels_span = []

        for sent in doc['content']:
            words.append(sent['tokens'])
            labels_sta.append(['O' for i in range(0, len(sent['tokens']))])
            labels_end.append(['O' for i in range(0, len(sent['tokens']))])
            candidates.append([0] * len(sent['tokens']))
            candidates_sta.append([0] * len(sent['tokens']))
            candidates_end.append([0] * len(sent['tokens']))
            candidates_span.append([])
            labels_span.append([])

        if mode != 'test':
            for event in doc['events']:
                for mention in event['mention']:
                    labels_sta[mention['sent_id']][mention['offset'][0]] = event['type']
                    labels_end[mention['sent_id']][mention['offset'][1] - 1] = event['type']
                    for k in range(mention['offset'][0], mention['offset'][1]):
                        candidates[mention['sent_id']][k] = 1
                    candidates_sta[mention['sent_id']][mention['offset'][0]] = 1
                    candidates_end[mention['sent_id']][mention['offset'][1] - 1] = 1
                    candidates_span[mention['sent_id']].append([mention['offset'][0], mention['offset'][1] - 1])
                    labels_span[mention['sent_id']].append(event['type'])
            for mention in doc['negative_triggers']:
                labels_sta[mention['sent_id']][mention['offset'][0]] = 'O'
                labels_sta[mention['sent_id']][mention['offset'][1] - 1] = 'O'
                for k in range(mention['offset'][0], mention['offset'][1]):
                    candidates[mention['sent_id']][k] = 1
                candidates_sta[mention['sent_id']][mention['offset'][0]] = 1
                candidates_end[mention['sent_id']][mention['offset'][1] - 1] = 1
                candidates_span[mention['sent_id']].append([mention['offset'][0], mention['offset'][1] - 1])
                labels_span[mention['sent_id']].append('O')

        elif mode == 'test':
            # pdb.set_trace()
            for mention in doc['candidates']:
                labels_sta[mention['sent_id']][mention['offset'][0]] = 'O'
                labels_sta[mention['sent_id']][mention['offset'][1] - 1] = 'O'
                for k in range(mention['offset'][0], mention['offset'][1]):
                    candidates[mention['sent_id']][k] = 1
                candidates_sta[mention['sent_id']][mention['offset'][0]] = 1
                candidates_end[mention['sent_id']][mention['offset'][1] - 1] = 1
                candidates_span[mention['sent_id']].append([mention['offset'][0], mention['offset'][1] - 1])
                labels_span[mention['sent_id']].append('O')
        for i in range(0, len(words)):
            record = {
                'guid': "%s-%s-%d" % (mode, doc['id'], i),
                'word': words[i],
                # 'labels_sta': labels_sta[i],
                # 'labels_end': labels_end[i],
                # 'candidates': candidates[i],
                # 'candidates_sta': candidates_sta[i],
                # 'candidates_end': candidates_end[i],
                'candidates_span': candidates_span[i],
                'labels_span': labels_span[i]
            }

            examples.append(record)
    write_json(dealed_data_dir, examples)
    return examples


def main_span():
    maven_train_path = './maven/train.jsonl'
    maven_valid_path = './maven/valid.jsonl'
    maven_test_path = './maven/test.jsonl'

    dealed_maven_train_path = './maven_processed/train.json'
    dealed_maven_valid_path = './maven_processed/valid.json'
    dealed_maven_test_path = './maven_processed/test.json'

    transform_data_into_span(maven_train_path, 'train', dealed_maven_train_path)
    transform_data_into_span(maven_valid_path, 'valid', dealed_maven_valid_path)
    transform_data_into_span(maven_test_path, 'test', dealed_maven_test_path)


if __name__ == '__main__':
    main_span()