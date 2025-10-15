import collections
import json
import joblib
import os
import re
import numpy as np
from functions import *
from tqdm import tqdm

ngram_len = 1 # Maximum n-gram length

def identify_tokens(songs, max_vocab_size):
    """
    Build a vocabulary of the most frequent character n-grams.

    - Counts all n-grams from length 1 to 5
    - Keeps only the top `num_ngrams`
    - Returns dictionaries for both token->id and id->token

    Parameters:
        songs (touple) : single touple containing touple which hold the string characters for a song
    """
    ngram_counts = collections.Counter()

    for song in tqdm(songs, desc=f"Creating and indexing top {max_vocab_size} n-grams"):
        length = len(song)
        for n in range(1, ngram_len + 1):  # n-grams of length 1 to 5
            for i in range(length - n + 1):
                ngram_counts[tuple(song[i:i + n])] += 1

    top_n = [token for token, _ in ngram_counts.most_common(max_vocab_size - 2)]
    token2idx = {token: idx for idx, token in enumerate(top_n)}
    token2idx[('^',)] = len(token2idx)
    token2idx[('$',)] = len(token2idx)
    idx2token = {idx: token for token, idx in token2idx.items()}
    return token2idx, idx2token


def tokenize_song(song, token2idx):
    """
    Convert text into a list of token IDs.

    Uses greedy longest-match-first tokenization with a max token length of 5.
    """

    tokens = []
    idx = 0
    while idx < len(song):
        for token_length in range(min(ngram_len, len(song) - idx), 0, -1):
            substring = tuple(song[idx: idx + token_length])
            if substring in token2idx:
                tokens.append(token2idx[substring])
                idx += token_length
                break
        else:
            # If no token matched (shouldn't usually happen), skip one char
            idx += 1
    return tokens


def tokenize_songs(songs, token2idx):
    """
    Tokenize a list of lines and return them sorted by length.

    Parameters:
        songs (list) :  for a song
    """
    tokenized_songs = [tokenize_song(['^'] + song + ["$"], token2idx) for song in tqdm(songs, desc="Tokenizing corpus")]
    tokenized_songs.sort(key=len)
    return tokenized_songs


def ngram_to_sentence(song, dicts):
  """Convert a sequence with ngrams to a sentence"""
  _, idx2tok = dicts
  sentence = []
  for ngram in song:
    sentence += list(idx2tok[ngram if isinstance(ngram, int) else ngram.item()])
  return sentence # Now a sentence


def process_corpus(songs_list, max_vocab_size, min_song_len):
    """
    Build the cleaned corpus from scratch:

    - Reads all .txt files in `corpus_dir`
    - Cleans text and splits into sentence-like lines
    - Builds vocabulary of top n-grams
    - Tokenizes lines
    - Saves results to JSON files
    """

    # load songs in a songs tuple and a songs list
    songs_tuple = tuple([tuple(song) for song in songs_list])

    # Build vocab and tokenize corpus
    token2idx, idx2token = identify_tokens(songs_tuple, max_vocab_size)
    tokens = tokenize_songs(songs_list, token2idx)
    # tokens_arr = np.array(tokens)

    # Make sure the songs are at least min_song_len events long
    tokens = list(filter(lambda x: len(x) >= min_song_len, tokens))

    # Save everything
    print("Saving corpus and dictionaries...")
    with open("corpus_tokenized.joblib", 'wb') as f:
        joblib.dump(tokens, f)
    with open("token2idx.joblib", 'wb') as f:
        joblib.dump(token2idx, f)
    with open("idx2token.joblib", 'wb') as f:
        joblib.dump(idx2token, f)

    return tokens, token2idx, idx2token


def get_cleaned_corpus(data_folder, max_vocab_size=10000, max_songs=float('inf'), min_song_len=1) -> tuple[list[list[int]],dict,dict]:
    """
    Load processed corpus from disk if available, otherwise generate it.
    """
    if all(os.path.exists(f) for f in ("token2idx.joblib", "idx2token.joblib", "corpus_tokenized.joblib")):
        with open("corpus_tokenized.joblib", 'rb') as f:
            corpus = joblib.load(f)
        with open("token2idx.joblib", 'rb') as f:
            token2idx = joblib.load(f)
        with open("idx2token.joblib", 'rb') as f:
            idx2token = {int(idx): token for idx, token in joblib.load(f).items()}
        return corpus, token2idx, idx2token

    if not data_folder:
        print("Missing data folder!")
        exit(2)
    print('Getting data...')
    songs_list = get_data(data_folder, max_songs)

    print('Processing corpus...')
    return process_corpus(songs_list, max_vocab_size, min_song_len)


def get_dictionaries(data_folder='', max_vocab_size=10000, max_songs=float('inf')) -> tuple[dict, dict]:
    """Load token dictionaries from disk if available, otherwise build them.
    Returns
    -------
    tok2idx : dict
        A dictionary that maps from a token str to an encoded number.
    idx2tok : dict
    """
    return get_cleaned_corpus(data_folder, max_vocab_size, max_songs)[1:]
