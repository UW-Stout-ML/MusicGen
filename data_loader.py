import torch
import numpy as np
from Song_corpus_generator import *
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader

tok2idx, idx2tok = get_dictionaries()

def collate_fn(batch: list[torch.tensor]):
  # enforce_sorted should be False if the dataset is not sorted
  # it will be sorted if it is set to False, but will be slower
  # return pack_sequence(batch, enforce_sorted=False)
  
  # Find the biggest song and fill the rest with the eos token
  eos_tok = tok2idx[('$',)]
  return pad_sequence(batch, batch_first=True, padding_value=eos_tok)

class Corpus(Dataset):
  def __init__(self, fileloc, vocab_size=10000, max_songs=float('inf')):
    self.midi_paths = []
    corpus_list = get_cleaned_corpus(fileloc, vocab_size, max_songs)[0]
    self.corpus = [torch.tensor(l) for l in corpus_list]

  def __len__(self):
    return len(self.corpus)
  
  def __getitem__(self, idx):
    return self.corpus[idx]
