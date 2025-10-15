from Song_corpus_generator import *

corpus, tok2idx, idx2tok = get_cleaned_corpus('')

print(len(tok2idx), tok2idx.keys())

# corpus_idx = 1
# corpus_song = corpus[corpus_idx]
# corpus_sentence = ngram_to_sentence(corpus_song, (tok2idx, idx2tok))

# print(corpus_sentence[:1000])