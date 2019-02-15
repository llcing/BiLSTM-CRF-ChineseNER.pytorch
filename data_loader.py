import numpy as np
import torch
from const import * 


class DataLoader(object):
    def __init__(self, sents, labels, cuda=True,
                 batch_size=64, shuffle=True, evaluation=False):
        self.cuda = cuda
        self.sents_size = len(sents)
        self._step = 0
        self._stop_step = self.sents_size // batch_size
        self.evaluation = evaluation

        self._batch_size = batch_size
        self._sents = np.asarray(sents)
        self._label = np.asarray(labels)

        if shuffle:
            self._shuffle()

    def _shuffle(self):
        indices = np.arange(self._sents.shape[0])
        np.random.shuffle(indices)
        self._sents = self._sents[indices]
        self._label = self._label[indices]

    def __iter__(self):
        return self

    def __next__(self):

        def pad_to_longest(insts):

            seq_len_list = [len(inst) for inst in insts]
            max_len = max(len(inst) for inst in insts)
            inst_data = np.array(
            [inst + [PAD] * (max_len - len(inst)) for inst in insts])

            if self.evaluation:
                with torch.no_grad():
                    inst_data_tensor = torch.from_numpy(inst_data)
            else:
                inst_data_tensor = torch.from_numpy(inst_data)

            
            inst_data_tensor = inst_data_tensor.cuda()
            seq_len = torch.LongTensor(seq_len_list).cuda()

            return inst_data_tensor, seq_len


        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        _bsz = self._batch_size
        self._step += 1

        word, seq_len = pad_to_longest(self._sents[_start:_start + _bsz])
        label, _ = pad_to_longest(self._label[_start:_start + _bsz])
        seq_len, perm_idx = seq_len.sort(0, descending=True)
        word = word[perm_idx]
        label = label[perm_idx]
        _, unsort_idx = perm_idx.sort(0, descending=False)

        return word, label, seq_len, unsort_idx
