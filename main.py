from Song_corpus_generator import *


data_folder = 'MusicRNN\\cleaned_midi'
<<<<<<< Updated upstream
max_songs = 10000
a, b = get_dictionaries(data_folder, max_songs)
# print(a, b)
=======
max_songs = float('inf')

# files = get_valid_midi_file_paths(data_folder)
# p = pm.PrettyMIDI(files[0])
# p = pretty_midi_to_piano_roll(p, 12, 96, 100)

# s = sentence_to_piano_roll(get_data(data_folder, max_songs=1)[0], 12, 96, 100)
# show_midi(sentence_to_piano_roll(s[0], 12, 96, 100))
# show_midi(sentence_to_piano_roll(s[0], 12, 96, 100))
# compare_midis(p, s)

a, b = get_dictionaries(data_folder, max_songs=max_songs)
>>>>>>> Stashed changes
