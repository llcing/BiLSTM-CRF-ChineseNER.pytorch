import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "START": 7, "STOP": 8
             }


def read_corpus(corpus_path, label_path, is_train=True):
    sents = []
    labels = []
    data = []
    
    with open(corpus_path, encoding='utf-8') as fr:
        lines_co = fr.readlines()
    with open(label_path) as fl:
        lines_lb = fl.readlines()
    
    if not is_train:
        word2id = read_dictionary('./dataset/vocab.pkl')
    else:
        word2id = {}

    for line_co, line_lb in zip(lines_co, lines_lb):
        sent_ = line_co.strip().split()
        tag_ = line_lb.strip().split()

        data.append((sent_, tag_))
        
        sentence_id = []

        for word in sent_:
            if is_train:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = len(word2id)+1
          
                sentence_id.append(word2id[word])
            
            else:
                if word.isdigit():
                    word = '<NUM>'
                elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word = '<UNK>'
                sentence_id.append(word2id[word])

        
        label_ = []
        for tag in tag_:
            label = tag2label[tag]
            label_.append(label)


        sents.append(sentence_id)
        labels.append(label_)


        
        
    if is_train: 
        # new_id = 1
        # for word in word2id.keys():
        #     word2id[word] = new_id
        #     new_id += 1
        word2id['<UNK>'] = len(word2id)+1
        word2id['<PAD>'] = 0

        print('vocabulary length:', len(word2id))
        with open('./dataset/vocab.pkl', 'wb') as fw:
            pickle.dump(word2id, fw)
   
    
    return sents, labels, len(word2id), data 




def read_dictionary(vocab_path):
    """

    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    # print('vocab_size:', len(word2id))
    return word2id



