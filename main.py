from Song_corpus_generator import *


data_folder = 'MusicRNN\\cleaned_midi'
max_songs = float('inf')
a, b = get_dictionaries(data_folder, max_songs=max_songs)
# print(a, b)
