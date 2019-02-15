import argparse


parser = argparse.ArgumentParser(description='LSTM_CRF')
parser.add_argument('--epochs', type=int, default=32,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--use-cuda', action='store_true',
                    help='enables cuda')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--use-crf', action='store_true',
                    help='use crf')

parser.add_argument('--mode', type=str, default='train',
                    help='train mode or test mode')

parser.add_argument('--save', type=str, default='./checkpoints/lstm_crf.pth',
                    help='path to save the final model')
parser.add_argument('--save-epoch', action='store_true',
                    help='save every epoch')
parser.add_argument('--data', type=str, default='dataset',
                    help='location of the data corpus')

parser.add_argument('--word-ebd-dim', type=int, default=300,
                    help='number of word embedding dimension')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='the probability for dropout')
parser.add_argument('--lstm-hsz', type=int, default=300,
                    help='BiLSTM hidden size')
parser.add_argument('--lstm-layers', type=int, default=2,
                    help='biLSTM layer numbers')
parser.add_argument('--l2', type=float, default=0.005,
                    help='l2 regularization')
parser.add_argument('--clip', type=float, default=.5,
                    help='gradient clipping')
parser.add_argument('--result-path', type=str, default='./result',
                    help='result-path')

args = parser.parse_args()

import torch 
torch.manual_seed(args.seed)
args.use_cuda = True



# load data
from data_loader import DataLoader
from data import read_corpus, tag2label
import os 
from eval import conlleval


sents_train, labels_train, args.word_size, _ = read_corpus(os.path.join('.', args.data, 'source_data.txt'), os.path.join('.', args.data, 'source_label.txt'))
sents_test, labels_test, _, data_origin = read_corpus(os.path.join('.', args.data, 'test_data.txt'), os.path.join('.', args.data, 'test_label.txt'), is_train=False)
args.label_size = len(tag2label)

train_data = DataLoader(sents_train, labels_train, cuda=args.use_cuda, batch_size=args.batch_size)
test_data = DataLoader(sents_test, labels_test, cuda=args.use_cuda, shuffle=False, evaluation=True, batch_size=args.batch_size)

from model import Model 
model = Model(args)


if args.use_cuda:
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.l2)

def train():
    model.train()
    total_loss = 0
    for word, label, seq_lengths, _  in train_data:
        optimizer.zero_grad()
        loss, _ = model(word, label, seq_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
    return total_loss / train_data._stop_step

def evaluate(epoch):
        model.eval()
        eval_loss = 0
        
        model_predict = []
        sent_res = []
        
        label2tag = {}
        for tag, lb in tag2label.items():
            label2tag[lb] = tag if lb != 0 else lb
        
        label_list = []

        for word, label, seq_lengths, unsort_idx in test_data:
            loss, _ = model(word, label, seq_lengths)
            pred = model.predict(word, seq_lengths)
            pred = pred[unsort_idx]
            seq_lengths = seq_lengths[unsort_idx]

            for i, seq_len in enumerate(seq_lengths.cpu().numpy()):
                pred_ = list(pred[i][:seq_len].cpu().numpy())
                label_list.append(pred_)
                
            eval_loss += loss.detach().item()

            
        for label_, (sent, tag) in zip(label_list, data_origin):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                # print(sent)
                print(len(sent))
                print(len(label_))
                # print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)

        label_path = os.path.join(args.result_path, 'label_' + str(epoch))
        metric_path = os.path.join(args.result_path, 'result_metric_' + str(epoch))
        
        for line in conlleval(model_predict, label_path, metric_path):
            print(line)
        
        return eval_loss / test_data._stop_step

import time 
train_loss = []
if args.mode == 'train':
    best_acc = None
    total_start_time = time.time()

    print('-' * 90)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss * 1000.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
                epoch, time.time() - epoch_start_time, loss))
        eval_loss = evaluate(epoch)
        torch.save(model.state_dict(), args.save)

    # TODO
    # if args.mode == 'test':
    #     test()

