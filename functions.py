import pretty_midi as pm
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from visuals import *

instrument_groups = {
  0: range(0, 8), # Piano → Acoustic Grand Piano
  11: range(8, 16), # Chromatic Percussion → Vibraphone
  17: range(16, 24), # Organ → Drawbar Organ
  27: range(24, 32), # Guitar → Electric Guitar (Clean)
  33: range(32, 40), # Bass → Electric Bass (Finger)
  41: range(40, 48), # Strings → Violin
  49: range(48, 56), # Ensemble → String Ensemble 1
  57: range(56, 64), # Brass → Trumpet
  65: range(64, 72), # Reed → Alto Sax
  73: range(72, 80), # Pipe → Flute
  81: range(80, 88), # Synth Lead → Lead 1 (Square)
  89: range(88, 96), # Synth Pad → Pad 2 (Warm)
  99: range(96, 104), # Synth Effects → FX 3 (Crystal)
  105: range(104, 112), # Ethnic → Sitar
  113: range(112, 120), # Percussive → Tinkle Bell
  125: range(120, 128), # Sound Effects → Helicopter
}

def get_instrument_group(program_number):
  for program, program_group in instrument_groups.items():
    if program_number in program_group:
      return program
  print("Error: Unknown program number")
  exit(3)

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
    start_vel = None # Keep track of the initial velocity of the note
    for column in range(piano_roll.shape[1]):
      current_time = column_to_second(column, fs)
      vel = piano_roll[semitone,column]

      if vel == 0:
        if start_time is None: continue # Skip empty notes
        note = pm.Note(velocity=start_vel, pitch=semitone, start=start_time, end=current_time)
        instrument.notes.append(note)
        start_time = None
        start_vel = None
      else:
        if start_time == None:
          start_time = current_time
          start_vel = int(vel)
    # Don't forget to add the last note
    if start_time is not None:
      current_time = column_to_second(piano_roll.shape[1], fs)
      note = pm.Note(velocity=start_vel, pitch=semitone, start=start_time, end=current_time)
      instrument.notes.append(note)
      start_time = None
      start_vel = None

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
      if time_step_count == 100:
        combined_sentence.append(f"ts({time_step_count})")
        time_step_count = 0
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
          # 32 velocities, max is 127
          vel = min(vel, 127) # Cap for safety
          bin_size = 127 / 32
          quantized_velocity = int(round(vel / bin_size) * bin_size)
          sentence.append(f"{note_idx}+")
          sentence.append(f"v{quantized_velocity}")
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

def sentence_to_piano_roll(sentence, min_note, max_note):
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
  note_velocities = {}
  last_note = None

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
        last_note = note
      continue
    elif token.endswith('-'):
      note = int(token[:-1])
      if note in active_notes:
        active_notes.remove(note)
      continue
    elif token.startswith('v'):
      vel = int(token[1:])
      note_velocities[last_note] = vel

    # Set the velocity for all active notes at the current time step
    # Notes turned on before this 'ts' are written into the previous step
    write_time_step = max(0, current_time_step - 1)
    for note in active_notes:
      if min_note <= note < max_note and write_time_step < length: # Ensure within bounds
        piano_roll[note - min_note, write_time_step] = note_velocities[note]

  # Add remaining active notes at the end
  write_time_step = max(0, current_time_step - 1)
  for note in active_notes:
    if min_note <= note < max_note and write_time_step < length: # Ensure within bounds
      piano_roll[note - min_note, write_time_step] = note_velocities[note]

  return piano_roll

# New method for parsing a PrettyMIDI object to a sentence and vice versa

def combined_instrument_events(pretty_midi: pm.PrettyMIDI, fs=100):
  # Concat all the notes from all the instruments into one list
  events = [] # (instrument, time, pitch, velocity, is_on)

  for inst in pretty_midi.instruments:
    if inst.is_drum: continue
    for note in inst.notes:
      if note.pitch > 127 or note.pitch < 0: continue
      start_time = int(note.start * fs)
      end_time = int(note.end * fs)
      events.append((inst.program, start_time, note.pitch, note.velocity, True))
      events.append((inst.program, end_time, note.pitch, note.velocity, False))
  
  return sorted(events, key=lambda x: x[1]) # Sort by time

def pretty_midi_to_sentence(pretty_midi: pm.PrettyMIDI, fs=100, combine_instruments=True) -> list[str]:
  """Skipping the piano roll step because we are using multiple instruments now.
  
  Parameters
  ----------
  pretty_midi : PrettyMIDI
    The PrettyMIDI object to convert to a sentence.
  min_note : int
    The minimum note (inclusive) in the piano roll.
  max_note : int
    The maximum note (exclusive) in the piano roll.
  fs : int
    The sampling frequency for the piano roll (notes/sec).

  Returns
  -------
  sentence : list[str]
    The sentence representation of the PrettyMIDI object.
  """
  
  def quantize_velocity(vel):
    max_vel = 127
    min_vel = 10
    num_bins = 32
    quantize = max_vel / num_bins
    new_vel = round(vel / quantize) * quantize
    new_vel = max(min_vel, new_vel)
    new_vel = min(max_vel, new_vel)
    return int(new_vel)

  events = combined_instrument_events(pretty_midi, fs)
  note_idx = 0
  last_time = 0
  sentence = []
  active_notes = set()
  note_played = False
  
  while note_idx < len(events):
    inst, time, pitch, vel, is_on = events[note_idx]
    note_idx += 1
    vel = quantize_velocity(vel)
    

    if combine_instruments:
      inst = get_instrument_group(inst)

    while time > last_time:
      diff = min(time - last_time, fs)
      last_time += diff
      if note_played:
        sentence += [f"t{diff}"]

    if is_on and (pitch, inst) not in active_notes:
      # sentence += [f"p{inst}"]
      # sentence += [f"v{vel}"]
      # sentence += [f"+{pitch}"]
      sentence += [f"v{vel}"]
      sentence += [f"+{pitch}p{inst}"]
      active_notes.add((pitch, inst))
      note_played = True
    elif not is_on and (pitch, inst) in active_notes:
      # sentence += [f"p{inst}"]
      # sentence += [f"-{pitch}"]
      sentence += [f"-{pitch}p{inst}"]
      active_notes.remove((pitch, inst))
  
  return sentence

def sentence_to_pretty_midi(sentence, fs=100):
  """Parse sentence to PrettyMIDI object.
  
  Parameters
  ----------
  sentence : list[str]
    The sentence to parse to a PrettyMIDI object.
  
  Returns
  -------
  pretty_midi : PrettyMIDI
    The PrettyMIDI object created from the sentence.
  """

  pretty_midi = pm.PrettyMIDI()
  instruments = {}
  active_notes = {}
  vel = 100
  inst = 0
  time = 0

  for idx, note in enumerate(sentence):
    if note[0] == 't': # Time step
      step = int(note[1:]) / fs
      time += step
    elif note[0] == 'v': # Update velocity
      vel = int(note[1:])
    # elif note[0] == 'p': # Update instrument
    #   inst = int(note[1:])
    elif note[0] == '+': # Note on
      idx = note.index('p')
      pitch = int(note[1:idx])
      inst = int(note[idx+1:])
      active_notes[(pitch, inst)] = (time, vel)
    elif note[0] == '-': # Note off
      idx = note.index('p')
      pitch = int(note[1:idx])
      inst = int(note[idx+1:])
      
      if (pitch, inst) not in active_notes:
        continue
      
      start_time, vel = active_notes.pop((pitch, inst))
      
      # Create instrument if not created
      if inst not in instruments:
        instruments[inst] = pm.Instrument(program=inst)
      
      note = pm.Note(velocity=vel, pitch=pitch, start=start_time, end=time)
      instruments[inst].notes.append(note)
    elif note == '^' or note == '$':
      continue
    else:
      print(f"Error: Unknown note {note}")
      exit(4)

  for inst in instruments.values():
    pretty_midi.instruments.append(inst)

  return pretty_midi

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

def get_data(input_dir, max_songs=float('inf'), fs=100):
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
    try:
      print(file_path)
      pretty_midi = pm.PrettyMIDI(file_path)
    except Exception as e:
      print(e)
      print("Error: Failed to load song")
    sentence = pretty_midi_to_sentence(pretty_midi, fs)
    sentences += [sentence]

  return sentences
