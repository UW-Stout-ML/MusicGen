import collections
# from tqdm import tqdm
import re
import os
import torch
from torch.utils.data import DataLoader, Dataset
import pretty_midi as pm
from visuals import *
import json

# temperature = float(input("Enter temperature: "))
fileloc = 'cleaned_midi'
n_notes = 128
hidden_size = 8
batch_size = 16
num_layers = 1
fs = 100

class MidiDataSet(Dataset):
  def __init__(self, fileloc, maxfiles=None):
    self.midi_paths = []
    for root, _, files in os.walk(fileloc):
      for file in files:
        if len(self.midi_paths) == maxfiles: break
        _, ext = os.path.splitext(file)
        if ext == '.mid':
          self.midi_paths.append(os.path.join(root, file))

  def __len__(self):
    return len(self.midi_paths)
  
  def __getitem__(self, idx):
    # Load midi from file path
    path = self.midi_paths[idx]
    midi_data = pm.PrettyMIDI(path)
    length = 16*10*fs
    piano_roll = midi_data.get_piano_roll(fs, pedal_threshold=1)[:,:length] > 0
    tens = torch.zeros((length, n_notes))
    tens[:piano_roll.shape[1]] = torch.tensor(piano_roll).permute(1, 0) # time, note
    return tens

def load_songs(data: Dataset) -> collections.Counter:
    """
    Takes in a data source, and gets all unique note combinations

    Parameters:
        data (file path) : The file to load the data from

    Returns:

    """
    notes_counts = collections.Counter()
    loader = DataLoader(data, shuffle=True, batch_size=1, pin_memory=True)
    #want to load every song from the loader, then every column from the song
    for song in loader:
        song = song.squeeze()
        # print(song.size())
        for col in song:
            col = tuple(col.tolist())
            notes_counts[col] += 1
    
    return notes_counts

def identify_tokens(songs, num_ngrams=5000):
    """
    Build a vocabulary of the most frequent note sequences n-grams.

    - Counts all note n-grams from length 1 to 4
    - Keeps only the top `num_ngrams`
    - Returns dictionaries for both token->id and id->token

    Parameters:
        songs (arrayLike) : a vector of notes being played at one time (every 1/16th of a second)
        num_ngrams (int) : Number of unique combinations to learn

    """
    ngram_counts = collections.Counter()

    for seq in tqdm(songs, desc=f"Creating and indexing top {num_ngrams} n-grams"):
        length = len(seq)
        for n in range(1, 5):  # n-grams of length 1 to 4
            for i in range(length - n + 1):
                ngram_counts[seq[i:i + n]] += 1

    top_n = [token for token, _ in ngram_counts.most_common(num_ngrams - 2)]
    token2idx = {token: idx for idx, token in enumerate(top_n)}


    #this will need to chage to find start and end of sequence
    token2idx['^'] = len(token2idx)
    token2idx['$'] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token

def tokenize(music, token2idx):
    """
    Convert text into a list of token IDs.

    Uses greedy longest-match-first tokenization with a max token length of 5.
    """
    tokens = []
    idx = 0
    while idx < len(music):
        for token_length in range(min(5, len(music) - idx), 0, -1):
            substring = music[idx: idx + token_length]
            if substring in token2idx:
                tokens.append(token2idx[substring])
                idx += token_length
                break
        else:
            # If no token matched (shouldn't usually happen), skip one char
            idx += 1
    return tokens

def tokenize_list(lines, token2idx):
    """
    Tokenize a list of lines and return them sorted by length.
    """
    tokenized_lines = [tokenize('^' + line + '$', token2idx) for line in tqdm(lines, desc="Tokenizing corpus")]
    tokenized_lines.sort(key=len)
    return tokenized_lines

data = MidiDataSet('cleaned_midis')

# for song in data:
#    show_midi_tensor(song)
#    break

counts = load_songs(data)

print("exporting!")

keys = list(counts.keys())
values = list(counts.values())
values = sorted(values, reverse=True)

# print(keys[0])
# print(keys[1])

print(f"Max count: {max(values)}")

with open('counts.json', 'w') as file:
    json.dump(values, file)
