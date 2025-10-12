import torch
import numpy as np
from Song_corpus_generator import *
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import random

def collate_fn(prob_sos, input_len, target_len):
  min_song_len = input_len + target_len + 1

  def collate(batch):
    sos_mode = random.random() < prob_sos
    inputs = torch.empty((len(batch), input_len), dtype=torch.long)
    targets = torch.empty((len(batch), target_len), dtype=torch.long)

    # For each song, slice it up!
    for i, song in enumerate(batch):
      song = batch[i] # A tensor!

      if sos_mode:
        # Tiny slice
        start_idx = 0
        mid_idx = 1
      else:
        # Big slice
        start_idx = random.randint(0, song.shape[0] - min_song_len - 1)
        mid_idx = start_idx + input_len
      
      end_idx = mid_idx + target_len
      input = song[start_idx:mid_idx]
      target = song[mid_idx:end_idx]
      
      inputs[i] = input
      targets[i] = target

    return inputs, targets

  return collate


class Corpus(Dataset):
  def __init__(
    self,
    data_path,
    input_len,
    target_len,
    max_corp_songs=float('inf'),
    max_songs=float('inf'),
    max_vocab_size=float('inf'),
  ):
    self.sos_mode = False
    self.max_input_len = input_len
    self.input_len = input_len
    self.target_len = target_len
    self.min_song_len = input_len + target_len + 1
    corpus_list, tok2idx, _ = get_cleaned_corpus(data_path, max_vocab_size, max_corp_songs, self.min_song_len)
    self.vocab_size = len(tok2idx)
    # print(tok2idx.keys())

    # This is our dataset of encoded sentences
    self.corpus = [torch.tensor(l) for l in corpus_list]
    self.corpus = list(filter(lambda t: t.shape[0] >= self.min_song_len, self.corpus))

    if max_songs is not None:
      self.corpus = self.corpus[:max_songs]

  def __len__(self):
    return len(self.corpus)

  def __getitem__(self, idx):
    song = self.corpus[idx]
    return song

    # start_idx = random.randint(0, song.shape[0] - self.min_song_len - 1)
    # mid_idx = start_idx + self.input_len
    
    # end_idx = mid_idx + self.target_len

    # input = song[start_idx:mid_idx]
    # target = song[mid_idx:end_idx]
    
    # return song, input, target