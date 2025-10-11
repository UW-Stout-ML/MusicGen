from Song_corpus_generator import *

corpus, _, _ = get_cleaned_corpus('MusicRNN\\cleaned_midi', 10000)

print(corpus[7])