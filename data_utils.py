import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR

def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{ROOT_DIR}/data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{ROOT_DIR}/data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels

def load_financial_phrasebank():
    train_sentences = []
    train_answers = []
    with open("./data/financial-phrasebank/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Sentence'])
            if myjson['Sentiment'] == 'positive':
                train_answers.append(0)
            elif myjson['Sentiment'] == 'negative':
                train_answers.append(1)
            elif myjson['Sentiment'] == 'neutral':
                train_answers.append(2)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/financial-phrasebank/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Sentence'])
            if myjson['Sentiment'] == 'positive':
                test_answers.append(0)
            elif myjson['Sentiment'] == 'negative':
                test_answers.append(1)
            elif myjson['Sentiment'] == 'neutral':
                test_answers.append(2)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_poem_sentiment():
    train_sentences = []
    train_answers = []
    with open("./data/poem-sentiment/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Verse text'])
            if myjson['Sentiment'] == 'negative':
                train_answers.append(0)
            elif myjson['Sentiment'] == 'positive':
                train_answers.append(1)
            elif myjson['Sentiment'] == 'no_impact':
                train_answers.append(2)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/poem-sentiment/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Verse text'])
            if myjson['Sentiment'] == 'negative':
                test_answers.append(0)
            elif myjson['Sentiment'] == 'positive':
                test_answers.append(1)
            elif myjson['Sentiment'] == 'no_impact':
                test_answers.append(2)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_subj():
    train_sentences = []
    train_answers = []
    with open("./data/subj/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Input'])
            if myjson['Label'] == 'objective':
                train_answers.append(0)
            elif myjson['Label'] == 'subjective':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/subj/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Input'])
            if myjson['Label'] == 'objective':
                test_answers.append(0)
            elif myjson['Label'] == 'subjective':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_civil_comments():
    train_sentences = []
    train_answers = []
    with open("./data/civil_comments/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Toxicity'] == 'neutral':
                train_answers.append(0)
            elif myjson['Toxicity'] == 'toxic':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/civil_comments/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Toxicity'] == 'neutral':
                test_answers.append(0)
            elif myjson['Toxicity'] == 'toxic':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_mr():
    train_sentences = []
    train_answers = []
    with open("./data/mr/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Review'])
            if myjson['Sentiment'] == 'negative':
                train_answers.append(0)
            elif myjson['Sentiment'] == 'positive':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/mr/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Review'])
            if myjson['Sentiment'] == 'negative':
                test_answers.append(0)
            elif myjson['Sentiment'] == 'positive':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_cr():
    train_sentences = []
    train_answers = []
    with open("./data/cr/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Review'])
            if myjson['Sentiment'] == 'negative':
                train_answers.append(0)
            elif myjson['Sentiment'] == 'positive':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/cr/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Review'])
            if myjson['Sentiment'] == 'negative':
                test_answers.append(0)
            elif myjson['Sentiment'] == 'positive':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def get_cb():
    train_questions = []
    train_answers = []
    with open(f"{ROOT_DIR}/data/cb/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            curr_label = myjson['label']
            if curr_label == 'contradiction':
                train_answers.append(0)
            elif curr_label == 'neutral':
                train_answers.append(1)
            elif curr_label == 'entailment':
                train_answers.append(2)
            # being a bit lazy here. We put the "question: " into the input and treat it like single sentence classification.
            train_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    test_questions = []
    test_answers = []
    with open(f"{ROOT_DIR}/data/cb/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'contradiction':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'entailment':
                test_answers.append(2)
            else:
                exit('answer')
            test_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    return train_questions, train_answers, test_questions, test_answers

def load_dbpedia():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_slot_movies(field_name):
    all_fields = ["Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
    assert field_name in all_fields
    all_fields.remove(field_name)
    filter_tags = [f"B-{field}" for field in all_fields] + [f"I-{field}" for field in all_fields] + ["O"]
    target_tags = [f"B-{field_name}", f"I-{field_name}"]

    with open(f'{ROOT_DIR}/data/slot-movies/train', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    train_answers = []
    train_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            train_answers.append(answer.strip())
            train_sentences.append(untagged_line.strip())

    with open(f'{ROOT_DIR}/data/slot-movies/test', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    test_answers = []
    test_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            test_answers.append(answer.strip())
            test_sentences.append(untagged_line.strip())

    return train_sentences, train_answers, test_sentences, test_answers

def load_atis(tag_name):
    with open(f'{ROOT_DIR}/data/atis/atis.train.pkl', 'rb') as stream:
        ds,dicts = pickle.load(stream)

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(ds.get, ['query', 'slot_labels', 'intent_labels'])

    tags_dict = {}
    train_sentences = []
    train_slot_strings = []
    for i in range(len(query)):
        slot_string = ''
        beginning_count = 0 # when there are multiple mentions of the destination city, we want to avoid those
        for j in range(len(query[i])):
            tag = i2s[slots[i][j]][2:]
            if tag in tags_dict.keys():
                tags_dict[tag] += 1
            else:
                tags_dict[tag] = 1

            if f'B-{tag_name}' in i2s[slots[i][j]]:
                beginning_count += 1
            if tag_name in i2s[slots[i][j]]:
                slot_string += i2t[query[i][j]] + ' '
        if slot_string != '' and beginning_count == 1:
            train_sentences.append(' '.join(map(i2t.get, query[i][1:-1]))) # [1:-1] cuts off BOS and EOS
            train_slot_strings.append(slot_string.strip())

    with open(f'{ROOT_DIR}/data/atis/atis.test.pkl', 'rb') as stream:
        ds,dicts = pickle.load(stream)

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(ds.get, ['query', 'slot_labels', 'intent_labels'])

    test_sentences = []
    test_slot_strings = []
    for i in range(len(query)):
        slot_string = ''
        beginning_count = 0 # when there are multiple mentions of the destination city, we want to avoid those
        for j in range(len(query[i])):
            if f'B-{tag_name}' in i2s[slots[i][j]]:
                beginning_count += 1
            if tag_name in i2s[slots[i][j]]:
                slot_string += i2t[query[i][j]] + ' '
        if slot_string != '' and beginning_count == 1:
            test_sentences.append(' '.join(map(i2t.get, query[i][1:-1]))) # [1:-1] cuts off BOS and EOS
            test_slot_strings.append(slot_string.strip())

    return train_sentences, train_slot_strings, test_sentences, test_slot_strings

def load_lama(which_lama):
    ### Load test data
    with open(f'{ROOT_DIR}/data/lama/original_rob/P{which_lama}/test.jsonl', 'r') as json_file:
        json_list = list(json_file)
    all_y_test = []
    all_x_test = []
    for json_str in json_list:
        result = json.loads(json_str)
        all_y_test.append(result['obj_label'])
        all_x_test.append(result['sub_label'])

    ### Load train data
    with open(f'{ROOT_DIR}/data/lama/original_rob/P{which_lama}/train.jsonl', 'r') as json_file:
        json_list = list(json_file)
    all_y_train = []
    all_x_train = []
    for json_str in json_list[:1000]:
        result = json.loads(json_str)
        all_y_train.append(result['obj_label'])
        all_x_train.append(result['sub_label'])

    with open(f'{ROOT_DIR}/data/lama/relations.jsonl', 'r') as json_file:
        json_list = list(json_file)
    template = None
    for json_str in json_list:
        result = json.loads(json_str)
        idx = int(result['relation'][1:])
        if idx == which_lama:
            template = result['template']
            x_pos = template.find('[X]')
            y_pos = template.find('[Y]')
            assert (x_pos >= 0) and (y_pos >= 0), "placeholder not found"
            if x_pos > y_pos:
                print("Not auto-regressive, skip")
                template = "INVALID"
            break

    return all_x_train, all_y_train, all_x_test, all_y_test, template

def load_rte():
    train_questions = []
    train_answers = []
    with open("data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open("data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    return train_questions, train_answers, test_questions, test_answers

def load_wnli():
    train_questions = []
    train_answers = []
    with open("data/wnli/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open("data/wnli/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    return train_questions, train_answers, test_questions, test_answers

def load_esnli():
    train_questions = []
    train_answers = []
    with open("data/esnli/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'entailment':
                train_answers.append(0)
            elif myjson['label'] == 'neutral':
                train_answers.append(1)
            elif myjson['label'] == 'contradiction':
                train_answers.append(2)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or Neutral or False?')

    test_questions = []
    test_answers = []
    with open("data/esnli/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'entailment':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'contradiction':
                test_answers.append(2)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or Neutral or False?')

    return train_questions, train_answers, test_questions, test_answers

def load_mnli():
    train_questions = []
    train_answers = []
    with open("data/mnli/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open("data/mnli/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    return train_questions, train_answers, test_questions, test_answers

def load_sick():
    train_questions = []
    train_answers = []
    with open("data/sick/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'entailment':
                train_answers.append(0)
            elif myjson['label'] == 'neutral':
                train_answers.append(1)
            elif myjson['label'] == 'contradiction':
                train_answers.append(2)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or Neutral or False?')

    test_questions = []
    test_answers = []
    with open("data/sick/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'entailment':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'contradiction':
                test_answers.append(2)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or Neutral or False?')

    return train_questions, train_answers, test_questions, test_answers

def load_gutenberg_time():
    train_sentences = []
    train_answers = []
    with open("./data/gutenberg_time/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'zero':
                train_answers.append(0)
            elif myjson['Label'] == 'one':
                train_answers.append(1)
            elif myjson['Label'] == 'two':
                train_answers.append(2)
            elif myjson['Label'] == 'three':
                train_answers.append(3)
            elif myjson['Label'] == 'four':
                train_answers.append(4)
            elif myjson['Label'] == 'five':
                train_answers.append(5)
            elif myjson['Label'] == 'six':
                train_answers.append(6)
            elif myjson['Label'] == 'seven':
                train_answers.append(7)
            elif myjson['Label'] == 'eight':
                train_answers.append(8)
            elif myjson['Label'] == 'night':
                train_answers.append(9)
            elif myjson['Label'] == 'ten':
                train_answers.append(10)
            elif myjson['Label'] == 'eleven':
                train_answers.append(11)
            elif myjson['Label'] == 'twelve':
                train_answers.append(12)
            elif myjson['Label'] == 'thirteen':
                train_answers.append(13)
            elif myjson['Label'] == 'fourteen':
                train_answers.append(14)
            elif myjson['Label'] == 'fifteen':
                train_answers.append(15)
            elif myjson['Label'] == 'sixteen':
                train_answers.append(16)
            elif myjson['Label'] == 'seventeen':
                train_answers.append(17)
            elif myjson['Label'] == 'eighteen':
                train_answers.append(18)
            elif myjson['Label'] == 'nineteen':
                train_answers.append(19)
            elif myjson['Label'] == 'twenty':
                train_answers.append(20)
            elif myjson['Label'] == 'twenty_one':
                train_answers.append(21)
            elif myjson['Label'] == 'twenty_two':
                train_answers.append(22)
            elif myjson['Label'] == 'twenty_three':
                train_answers.append(23)  
            else:
                exit('answer')
                
    test_sentences = []
    test_answers = []
    with open("./data/gutenberg_time/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'zero':
                test_answers.append(0)
            elif myjson['Label'] == 'one':
                test_answers.append(1)
            elif myjson['Label'] == 'two':
                test_answers.append(2)
            elif myjson['Label'] == 'three':
                test_answers.append(3)
            elif myjson['Label'] == 'four':
                test_answers.append(4)
            elif myjson['Label'] == 'five':
                test_answers.append(5)
            elif myjson['Label'] == 'six':
                test_answers.append(6)
            elif myjson['Label'] == 'seven':
                test_answers.append(7)
            elif myjson['Label'] == 'eight':
                test_answers.append(8)
            elif myjson['Label'] == 'night':
                test_answers.append(9)
            elif myjson['Label'] == 'ten':
                test_answers.append(10)
            elif myjson['Label'] == 'eleven':
                test_answers.append(11)
            elif myjson['Label'] == 'twelve':
                test_answers.append(12)
            elif myjson['Label'] == 'thirteen':
                test_answers.append(13)
            elif myjson['Label'] == 'fourteen':
                test_answers.append(14)
            elif myjson['Label'] == 'fifteen':
                test_answers.append(15)
            elif myjson['Label'] == 'sixteen':
                test_answers.append(16)
            elif myjson['Label'] == 'seventeen':
                test_answers.append(17)
            elif myjson['Label'] == 'eighteen':
                test_answers.append(18)
            elif myjson['Label'] == 'nineteen':
                test_answers.append(19)
            elif myjson['Label'] == 'twenty':
                test_answers.append(20)
            elif myjson['Label'] == 'twenty_one':
                test_answers.append(21)
            elif myjson['Label'] == 'twenty_two':
                test_answers.append(22)
            elif myjson['Label'] == 'twenty_three':
                test_answers.append(23)  
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_creak():
    train_sentences = []
    train_answers = []
    with open("./data/creak/train.json", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['sentence'])
            if myjson['label'] == 'false':
                train_answers.append(0)
            elif myjson['label'] == 'true':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/creak/dev.json", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['sentence'])
            if myjson['label'] == 'false':
                test_answers.append(0)
            elif myjson['label'] == 'true':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_sbic():
    train_sentences = []
    train_answers = []
    with open("./data/sbic/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Post'])
            if myjson['Offensive'] == 'neutral':
                train_answers.append(0)
            elif myjson['Offensive'] == 'offensive':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/sbic/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Post'])
            if myjson['Offensive'] == 'neutral':
                test_answers.append(0)
            elif myjson['Offensive'] == 'offensive':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_hate_speech18():
    train_sentences = []
    train_answers = []
    with open("./data/hate-speech18/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/hate-speech18/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_tweet_hate():
    train_sentences = []
    train_answers = []
    with open("./data/tweet-hate/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/tweet-hate/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_tweet_irony():
    train_sentences = []
    train_answers = []
    with open("./data/tweet-irony/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'ironic':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/tweet-irony/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'ironic':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_tweet_offensive():
    train_sentences = []
    train_answers = []
    with open("./data/tweet-offensive/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'offensive':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/tweet-offensive/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'offensive':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_tweet_atheism():
    train_sentences = []
    train_answers = []
    with open("./data/tweet-stance_atheism/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'none':
                train_answers.append(0)
            elif myjson['Label'] == 'against':
                train_answers.append(1)
            elif myjson['Label'] == 'favor':
                train_answers.append(2)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/tweet-stance_atheism/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'none':
                test_answers.append(0)
            elif myjson['Label'] == 'against':
                test_answers.append(1)
            elif myjson['Label'] == 'favor':
                test_answers.append(2)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_tweet_feminist():
    train_sentences = []
    train_answers = []
    with open("./data/tweet-stance_feminist/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'none':
                train_answers.append(0)
            elif myjson['Label'] == 'against':
                train_answers.append(1)
            elif myjson['Label'] == 'favor':
                train_answers.append(2)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/tweet-stance_feminist/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Tweet'])
            if myjson['Label'] == 'none':
                test_answers.append(0)
            elif myjson['Label'] == 'against':
                test_answers.append(1)
            elif myjson['Label'] == 'favor':
                test_answers.append(2)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_binary():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-binary/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-binary/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_race():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-race/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-race/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_religion():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-religion/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-religion/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_national_origin():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-national_origin/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-national_origin/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_gender():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-gender/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-gender/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_violence():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-violence/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-violence/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_ethos_disability():
    train_sentences = []
    train_answers = []
    with open("./data/ethos-disability/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            train_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                train_answers.append(0)
            elif myjson['Label'] == 'hate':
                train_answers.append(1)
            else:
                exit('answer')

    test_sentences = []
    test_answers = []
    with open("./data/ethos-disability/test.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            test_sentences.append(myjson['Text'])
            if myjson['Label'] == 'neutral':
                test_answers.append(0)
            elif myjson['Label'] == 'hate':
                test_answers.append(1)
            else:
                exit('answer')
    return train_sentences, train_answers, test_sentences, test_answers

def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
        params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'financial-phrasebank':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_financial_phrasebank()
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['positive'], 1: ['negative'], 2: ['neutral']}
        params['inv_label_dict'] = {'positive': 0, 'negative': 1, 'neutral': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'poem-sentiment':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_poem_sentiment()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Verse text: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['negative'], 1: ['positive'], 2: ['neutral']}
        params['inv_label_dict'] = {'negative': 0, 'positive': 1, 'neutral': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'civil_comments':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_civil_comments()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Toxicity: "
        params['label_dict'] = {0: ['neutral'], 1: ['toxic']}
        params['inv_label_dict'] = {'neutral': 0, 'toxic': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'subj':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_subj()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Input: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['objective'], 1: ['subjective']}
        params['inv_label_dict'] = {'objective': 0, 'subjective': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'mr':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mr()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'cr':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_cr()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'wnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_wnli()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'mnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mnli()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'esnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_esnli()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['True'], 1: ['neutral'], 2: ['False']}
        params['inv_label_dict'] = {'True': 0, 'neutral': 1, 'False': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'cb':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = get_cb()
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['false'], 1: ['neither'], 2: ['true']}
        params['inv_label_dict'] = {'false': 0, 'neither': 1, 'true': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'sick':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sick()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['True'], 1: ['neutral'], 2: ['False']}
        params['inv_label_dict'] = {'True': 0, 'neutral': 1, 'False': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'creak':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_creak()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Claim: "
        params["a_prefix"] = "Factuality: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'sbic':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sbic()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Post: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['offensive']}
        params['inv_label_dict'] = {'neutral': 0, 'offensive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet-hate':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_hate()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'tweet-irony':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_irony()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['ironic']}
        params['inv_label_dict'] = {'neutral': 0, 'ironic': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'tweet-offensive':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_offensive()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['offensive']}
        params['inv_label_dict'] = {'neutral': 0, 'offensive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'tweet-atheism':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_atheism()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['none'], 1: ['against'], 2: ['favor']}
        params['inv_label_dict'] = {'none': 0, 'against': 1, 'favor': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'tweet-feminist':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_tweet_feminist()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Tweet: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['none'], 1: ['against'], 2: ['favor']}
        params['inv_label_dict'] = {'none': 0, 'against': 1, 'favor': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'hate-speech18':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_hate_speech18()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'gutenbert_time':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_gutenberg_time()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = ": "
        params['label_dict'] = {0: ['0'], 1: ['1'], 2: ['2'], 3: ['3'], 4: ['4'], 5: ['5'], 6: ['6'], 7: ['7'], 8: ['8'], 9: ['9'], 10: ['10'], 11: ['11'], 12: ['12'], 13: ['13'], 14: ['14'], 15: ['15'], 16: ['16'], 17: ['17'], 18: ['18'], 19: ['19'], 20: ['20'], 21: ['21'], 22: ['22'], 23: ['23']}
        params['inv_label_dict'] = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10', 11: '11', 12: '12', 13: '13', 14: '14', 15: '15', 16: '16', 17: '17', 18: '18', 19: '19', 20: '20', 21: '21', 22: '22', 23: '23'}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ethos-binary':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_binary()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'ethos-race':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_race()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'ethos-religion':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_religion()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'ethos-national_origin':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_national_origin()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'ethos-gender':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_gender()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'ethos-violence':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_violence()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'ethos-disability':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_ethos_disability()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Text: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['neutral'], 1: ['hate']}
        params['inv_label_dict'] = {'neutral': 0, 'hate': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'dbpedia':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'][:4] == 'lama':
        which_lama = int(params['dataset'].split('_')[-1])
        all_x_train, all_y_train, all_x_test, all_y_test, template = load_lama(which_lama)

        # reject if template is not valid
        if template == "INVALID":
            params['template'] = template
            return None, None, None, None

        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = all_x_train, all_y_train, all_x_test, all_y_test
        params['prompt_prefix'] = ""
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1
        params['template'] = template

        x_pos = template.find('[X]')
        y_pos = template.find('[Y]')
        seg1 = template[0:x_pos]
        seg2 = template[x_pos+3:y_pos]

        def single_prompt_func(entity, target):
            return f"{seg1}{entity}{seg2}{target}"

        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            assert seg2[-1] == " "
            prompt = ""
            for x, y in zip(train_sentences, train_labels):
                prompt += single_prompt_func(x, y)
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{seg1}{test_sentence}{seg2}"[:-1]
            else:
                prompt += f"{seg1}{test_sentence}{seg2}"[:-1] + test_label_option
            return prompt

        example = single_prompt_func(orig_train_sentences[0], orig_train_labels[0])
        print(f"Sentence example: ||{example}||")

        params['prompt_func'] = prompt_func
        params['single_prompt_func'] = single_prompt_func

    elif params['dataset'][:9] == 'mit_movie':
        field_name = params['dataset'][10:]
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_slot_movies(field_name)
        """
        Actor 944
        Award 54
        Character_Name 225
        Director 415
        Genre 780
        Opinion 190
        Origin 178
        Plot 1459
        Quote 43
        Relationship 147
        Soundtrack 7
        Year 655
        """

        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = f"{field_name}: "
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1


        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            q_prefix = params["q_prefix"]
            a_prefix = params["a_prefix"]

            prompt = params['prompt_prefix']
            for x, y in zip(train_sentences, train_labels):
                prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
            else:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
            return prompt

        params['prompt_func'] = prompt_func

    elif params['dataset'][:4] == 'atis':
        tag_name = params['dataset'][5:]
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_atis(tag_name)

        name2prefix = {
            "airline_name": "Airline name",
            "depart_time.period_of_day": "Depart time - Period of day",
            "depart_date.day_name": "Depart date - Day name"
        }

        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = f"{name2prefix[tag_name]}: "
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1

        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            q_prefix = params["q_prefix"]
            a_prefix = params["a_prefix"]

            prompt = params['prompt_prefix']
            for x, y in zip(train_sentences, train_labels):
                prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
            else:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
            return prompt

        params['prompt_func'] = prompt_func

    else:
        raise NotImplementedError
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels