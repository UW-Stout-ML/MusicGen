import pretty_midi as pm
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from visuals import *

def clean_all_midis(input_dir):
  input_files = get_valid_midi_file_paths(input_dir)
  files_names = file_paths_to_files(input_files)

  for idx, file in enumerate(input_files):
    try:
      midi = pm.PrettyMIDI(file)
      full_path = os.path.join('cleaned_midis', f"{files_names[idx]}.mid")
      midi.write(full_path)
    except:
      print(f"Failed to load midi {idx}")

def validate_midi_file_path(file_path, warn=False):
  root, ext = os.path.splitext(file_path)
  if ext.lower() != '.mid':
    if warn: print(f"File {file_path} is invalid!")
  return True

def get_valid_midi_file_paths(input_dir, max_songs=float('inf')):
  valid_file_paths = []

  for root, _, files in os.walk(input_dir):
    for file in files:
      if len(valid_file_paths) >= max_songs: break
      if not validate_midi_file_path(file): continue
      full_file_path = os.path.join(root, file)
      valid_file_paths.append(full_file_path)
  
  return valid_file_paths

def pretty_midi_to_piano_roll(midi: pm.PrettyMIDI, min_note, max_note, fs):
  piano_roll = midi.get_piano_roll(fs=fs) # resolution is 1./fs seconds
  piano_roll = piano_roll[min_note:max_note]
  return piano_roll

def column_to_second(column: int, fs):
  return column / fs

def export_piano_roll(piano_roll: np.ndarray, dir, name, min_note, max_note, fs):
  # piano_roll: (notes, length)

  # Pad missing octaves with zeros
  new_piano_roll = np.zeros((128, piano_roll.shape[1]))
  new_piano_roll[min_note:max_note] = piano_roll
  piano_roll = new_piano_roll

  # Create a PrettyMIDI object from the piano roll
  midi = pm.PrettyMIDI()
  instrument = pm.Instrument(program=0)
  
  for semitone in range(piano_roll.shape[0]):
    start_time = None # Keep track of the start time of the note
    for column in range(piano_roll.shape[1]):
      current_time = column_to_second(column, fs)
      vel = piano_roll[semitone,column]

      if vel == 0:
        if start_time is None: continue # Skip empty notes
        note = pm.Note(velocity=100, pitch=semitone, start=start_time, end=current_time)
        instrument.notes.append(note)
        start_time = None
      else:
        if start_time == None:
          start_time = current_time
    # Don't forget to add the last note
    if start_time is not None:
      current_time = column_to_second(piano_roll.shape[1], fs)
      note = pm.Note(velocity=100, pitch=semitone, start=start_time, end=current_time)
      instrument.notes.append(note)
      start_time = None

  midi.instruments.append(instrument)

  # Create the directory if it doesn't exist
  if not os.path.exists(dir):
    os.makedirs(dir)

  full_path = os.path.join(dir, f"{name}.mid")
  midi.write(full_path)

def combine_time_steps(sentence):
  """Combine consecutive time steps in the sentence.

  Parameters
  ----------
  sentence : list of str
    The sentence to combine time steps in.
  
  Returns
  -------
  combined_sentence : list of str
    The sentence with combined time steps.
  """

  combined_sentence = []
  time_step_count = 0

  for token in sentence:
    if token == 'ts':
      time_step_count += 1
    else:
      if time_step_count > 0:
        combined_sentence.append(f"ts({time_step_count})")
        time_step_count = 0
      combined_sentence.append(token)

  # If there are remaining time steps at the end, add them
  if time_step_count > 0:
    combined_sentence.append(f"ts({time_step_count})")

  return combined_sentence

def uncombine_time_steps(sentence):
  """Uncombine time steps in the sentence.

  Parameters
  ----------
  sentence : list of str
    The sentence to uncombine time steps in.
  
  Returns
  -------
  uncombined_sentence : list of str
    The sentence with uncombined time steps.
  """

  uncombined_sentence = []

  for token in sentence:
    if token.startswith('ts(') and token.endswith(')'):
      count = int(token[3:-1])
      uncombined_sentence.extend(['ts'] * count)
    else:
      uncombined_sentence.append(token)

  return uncombined_sentence

def piano_roll_to_sentence(piano_roll: np.ndarray, min_note, max_note):
  """Get the sentence from a piano roll.

  Parameters
  ----------
  piano_roll : np.ndarray
    The piano roll to convert to a sentence. Shape (notes, length).
  min_note : int
    The minimum note (inclusive) in the piano roll.
  max_note : int
    The maximum note (exclusive) in the piano roll.
  
  Returns
  -------
  sentence : list of str
    The sentence representing the piano roll.
    ^ is the beginning of the sentence.
    $ is the end of the sentence.
    ts(n) is a time step of n/fs seconds.
    n+ is a note on event for note n.
    n- is a note off event for note n.
  """

  # Assert max_note - min_note == piano_roll.shape[0]
  assert max_note - min_note == piano_roll.shape[0], "Piano roll shape does not match min and max notes."

  sentence = []
  active_notes = set() # Keep track of active notes

  # sentence.append('^') # Beginning of the sentence

  piano_roll = piano_roll.T # Shape (length, notes)

  # Iterate over each time step (column)
  for col_idx, column in enumerate(piano_roll):
    # Iterate over each note (row)
    for note_idx, vel in enumerate(column):
      note_idx += min_note # Adjust note index to actual MIDI note number
      if vel > 0: # Note on
        if note_idx not in active_notes:
          sentence.append(f"{note_idx}+")
          active_notes.add(note_idx)
      else: # Note off
        if note_idx in active_notes:
          sentence.append(f"{note_idx}-")
          active_notes.remove(note_idx)

    # Add a time step after processing all notes in the column if not the last column
    if col_idx < piano_roll.shape[0] - 1:
      sentence.append(f"ts")

  # Turn off any remaining active notes at the end
  for note_idx in active_notes:
    sentence.append(f"{note_idx}-")

  # sentence.append('$') # End of the sentence

  return combine_time_steps(sentence)

def sentence_to_piano_roll(sentence, min_note, max_note, velocity=100):
  """Convert a sentence back to a piano roll.

  Parameters
  ----------
  sentence : list of str
    The sentence to convert to a piano roll.
    Contains tokens from piano_roll_to_sentence.
  min_note : int
    The minimum note (inclusive) in the piano roll.
  max_note : int
    The maximum note (exclusive) in the piano roll.
  velocity : int
    The velocity to use for note on events.
  """
  
  sentence = uncombine_time_steps(sentence)

  # First, determine the length of the piano roll
  length = 0
  for token in sentence:
    if token == 'ts':
      length += 1

  piano_roll = np.zeros((max_note - min_note, length))

  current_time_step = 0
  active_notes = set()

  for token in sentence:
    if token == '^':
      continue
    elif token == '$':
      continue
    elif token == 'ts':
      current_time_step += 1
    elif token.endswith('+'):
      note = int(token[:-1])
      if min_note <= note < max_note:
        active_notes.add(note)
      continue
    elif token.endswith('-'):
      note = int(token[:-1])
      if note in active_notes:
        active_notes.remove(note)
      continue

    # Set the velocity for all active notes at the current time step
    # Notes turned on before this 'ts' are written into the previous step
    write_time_step = max(0, current_time_step - 1)
    for note in active_notes:
      if min_note <= note < max_note and write_time_step < length: # Ensure within bounds
        piano_roll[note - min_note, write_time_step] = velocity

  # Add remaining active notes at the end
  write_time_step = max(0, current_time_step - 1)
  for note in active_notes:
    if min_note <= note < max_note and write_time_step < length: # Ensure within bounds
      piano_roll[note - min_note, write_time_step] = velocity

  return piano_roll

# Vectorized functions
def pretty_midis_to_piano_rolls(pretty_midis, min_note, max_note, fs):
  return [pretty_midi_to_piano_roll(midi, min_note, max_note, fs) for midi in pretty_midis]

def file_paths_to_pretty_midis(file_paths):
  midis = []
  
  for idx, file_path in enumerate(file_paths):
    try:
      midi = pm.PrettyMIDI(file_path)
      midis.append(midi)
    except:
      print(f"Failed to load midi {idx}")

  return midis

def export_piano_rolls(piano_rolls, dir, file_names, min_note, max_note, fs):
  for idx, piano_roll in enumerate(piano_rolls):
    file_name = file_names[idx] if file_names else f"song_{idx}"
    export_piano_roll(piano_roll, dir, file_name, min_note, max_note, fs)

def file_paths_to_files(file_paths):
  file_names = []
  
  for file_path in file_paths:
    # Everything after the last slash and before the extension is the file name
    file_name = file_path.split('\\')[-1]
    file_name = file_name.split('.')[0]
    file_names.append(file_name)

  return file_names

def piano_rolls_to_sentences(piano_rolls, min_note, max_note):
  return [piano_roll_to_sentence(piano_roll, min_note, max_note) for piano_roll in piano_rolls]

def sentences_to_piano_rolls(sentences, min_note, max_note):
  return [sentence_to_piano_roll(sentence, min_note, max_note) for sentence in sentences]

# Get data function

def get_data(input_dir, max_songs=float('inf'), min_note=12, max_note=96, fs=100):
  """Get a list of sentences (list of str) from MIDI files in the input directory.
  
  Parameters
  ----------
  input_dir : str
    The directory containing MIDI files.
  max_songs : int
    The maximum number of songs to process.
  min_note : int
    The minimum note (inclusive) to consider.
  max_note : int
    The maximum note (exclusive) to consider.
  fs : int
    The sampling frequency for the piano roll (notes/sec).
  """

  midi_file_paths = get_valid_midi_file_paths(input_dir, max_songs)
  sentences = []

  for file_path in tqdm(midi_file_paths, desc=f"Creating {len(midi_file_paths)} sentences"):
    pretty_midi = pm.PrettyMIDI(file_path)
    piano_roll = pretty_midi_to_piano_roll(pretty_midi, min_note, max_note, fs)
    sentence = piano_roll_to_sentence(piano_roll, min_note, max_note)
    sentences += [sentence]

  return sentences
