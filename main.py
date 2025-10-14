from Song_corpus_generator import *

data_folder = ''
max_songs = 10

songs, a, b = get_cleaned_corpus(data_folder, max_vocab_size=10000)

sentence = ngram_to_sentence(songs[529], (a, b))
print(ngram_to_sentence(songs[529], (a, b)))

pretty_midi = sentence_to_pretty_midi(sentence, 100)
show_pretty_midi(pretty_midi)
directory_path = 'output_songs'
file_count = len([entry for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])

file_path = os.path.join(directory_path, f"song_{str(file_count)}.mid")
pretty_midi.write(file_path)

# for idx, song in enumerate(tqdm(songs, desc='Finding repeats...')):
#     s = ngram_to_sentence(song, (a, b))
#     stop = False
#     k = 0
#     for i in range(1, len(s)):
#         if s[i] == s[i-1] and s[i][0] == 't':
#             k += 1
#             if k > 10:
#                 print(f"REPEAT!!! Song: {idx}")
#                 stop = True
#                 break
#         else:
#             k = 0
#     if stop: break

# sentence = ngram_to_sentence(songs[-1])
# print(sentence)
# pretty_midi = sentence_to_pretty_midi(sentence)
# pretty_midi.write('test.mid')

# file = get_valid_midi_file_paths(data_folder)[0]
# file = "C:\\Users\\Kyler\\Documents\\GitHub\\MusicGen\\non_piano_music\\vg_music_database\\vg_music_database\\microsoft\\xbox360\\LostOdyssey\\LO_A-Formidable-Enemy-Appears.mid"
# pretty_midi = pm.PrettyMIDI(file)

# print(pretty_midi.instruments)

# sentence = pretty_midi_to_sentence(pretty_midi)
# pretty_midi_2 = sentence_to_pretty_midi(sentence)

# pretty_midi_2.write('test.mid')

# show_pretty_midi(pretty_midi_2)

# print(sentence[:1000])

# a, b = get_dictionaries(data_folder, max_songs=max_songs)
