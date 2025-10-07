import torch
import numpy as np
from Song_corpus_generator import *
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import random

tok2idx, idx2tok = get_dictionaries()

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
    song_len = random.randint(target_len + 1, full_song_len)  
    
    return (
      batch[:,:song_len-target_len],
      batch[:,song_len-target_len:]
    )

  return collate_fn

class Corpus(Dataset):
  def __init__(
    self,
    fileloc,
    vocab_size=10000,
    max_songs=float('inf'),
    rand=False,
    target_len=2,
    cap=256
  ):
    self.midi_paths = []
    self.target_len=target_len
    self.rand = rand
    self.cap = cap
    corpus_list = get_cleaned_corpus(fileloc, vocab_size, max_songs)[0]
    self.corpus = [torch.tensor(l) for l in corpus_list][:max_songs]

  def __len__(self):
    return len(self.corpus)
  
  def __getitem__(self, idx):
    # Cap the number of n-grams in a song
    song = self.corpus[idx]
    song = song[:self.cap]
    if (self.rand):
      full_song_len = song.shape[0]
      # print(full_song_len, self.target_len, song)
      song_len = random.randint(self.target_len + 1, full_song_len)  
      return song[:song_len]
    return song
