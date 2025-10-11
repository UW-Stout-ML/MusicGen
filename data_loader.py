import torch
import numpy as np
from Song_corpus_generator import *
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import random

def get_collate_fn(target_len):
  def collate_fn(batch: list[torch.tensor]):
    # enforce_sorted should be False if the dataset is not sorted
    # it will be sorted if it is set to False, but will be slower
    # return pack_sequence(batch, enforce_sorted=False)
    # eos_tok = tok2idx[('$',)]
    # Find the biggest song and fill the rest with the eos token
    # return pad_sequence(batch, batch_first=True, padding_value=eos_tok)
    
    min_len = min([t.shape[0] for t in batch])
    batch = [t[:min_len] for t in batch]
    batch = torch.stack(batch, dim=0)

    # Random pizzas
    full_song_len = min_len
    # song_len = random.randint(target_len + 1, full_song_len)  
    song_len = 40

    output = (
      batch[:,:song_len-target_len],
      batch[:,song_len-target_len:song_len]
    )

    # print('collate', song_len, target_len, min_len)

    return output

  return collate_fn

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
    self.input_len = input_len
    self.target_len = target_len
    self.min_song_len = input_len + target_len + 1
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
