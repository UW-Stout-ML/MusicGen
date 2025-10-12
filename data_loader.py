import torch
import numpy as np
from Song_corpus_generator import *
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import random

class Corpus(Dataset):
  def __init__(
    self,
    data_path,
    input_len,
    target_len,
    max_corp_songs=float('inf'),
    max_songs=float('inf'),
    max_vocab_size=float('inf'),
    min_song_len=None,
  ):
    self.max_input_len = input_len
    self.input_len = input_len
    self.target_len = target_len
    self.min_song_len = input_len + target_len + 1 if min_song_len is None else min_song_len
    corpus_list, tok2idx, _ = get_cleaned_corpus(data_path, max_vocab_size, max_corp_songs, self.min_song_len)
    self.vocab_size = len(tok2idx)

    # This is our dataset of encoded sentences
    self.corpus = [torch.tensor(l) for l in corpus_list]
    if max_songs is not None:
      self.corpus = self.corpus[:max_songs]

  def __len__(self):
    return len(self.corpus)
  
  def __getitem__(self, idx):
    song = self.corpus[idx]
    start_idx = random.randint(0, song.shape[0] - self.min_song_len - 1)
    mid_idx = start_idx + self.input_len
    end_idx = mid_idx + self.target_len
    input = song[start_idx:mid_idx]
    target = song[mid_idx:end_idx]
    return input, target

  def randomize_input_size(self):
    self.input_len = random.randint(1, self.max_input_len)
