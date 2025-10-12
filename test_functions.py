from functions import *

min_note = 12
max_note = 96
fs = 100

file = get_valid_midi_file_paths('non_piano_music')[0]
pretty_midi = pm.PrettyMIDI(file)

print(pretty_midi.instruments)

# piano_roll = pretty_midi_to_piano_roll(pretty_midi, min_note, max_note, fs)
# sentence = piano_roll_to_sentence(piano_roll, min_note, max_note)
# piano_roll_2 = sentence_to_piano_roll(sentence, min_note, max_note)

# print(sentence)
# exit()
# compare_midis(piano_roll, piano_roll_2)
# show_midi(piano_roll_2)

# export_piano_roll(piano_roll_2, 'output_test', 'test', min_note, max_note, fs)