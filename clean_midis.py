import pretty_midi as pm
import os
from tqdm import tqdm

melodies_inst_names = (
  'melody',
  'meldoy',
  'melopy',
)

def get_midis(input_dir):
  valid_file_paths = []
  for root, _, files in os.walk(input_dir):
    for file in files:
      if not file.endswith('.mid'): continue
      full_file_path = os.path.join(root, file)
      valid_file_paths.append(full_file_path)
  return valid_file_paths

def clean_pretty_midi(pretty_midi: pm.PrettyMIDI):
  max_vel = 127
  top_vel = 0
  min_time = float('inf')
  
  # Remove drums
  for instrument in pretty_midi.instruments:
    if instrument.is_drum:
      pretty_midi.instruments.remove(instrument)

  # Find max velocity and start of first note
  for instrument in pretty_midi.instruments:
    for note in instrument.notes:
      top_vel = max(top_vel, note.velocity)
      min_time = min(min_time, note.start)

  # Normalize velocities and remove silence at the beginning
  for instrument in pretty_midi.instruments:
    for note in instrument.notes:
      note.velocity = int(note.velocity / max_vel * max_vel)
      note.start -= min_time
      note.end -= min_time
  
  return pretty_midi

input_dir = 'POP9092'
output_dir = 'POP9092_cleaned'
midis = get_midis(input_dir)

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

removed_cnt = 0
for idx, midi in enumerate(tqdm(midis, 'Cleaning midis')):
  file_name = midi.split('\\')[-1].split('.')[0]
  pretty_midi = pm.PrettyMIDI(midi)
  if len(pretty_midi.instruments) == 0:
    removed_cnt += 1
    continue
  if pretty_midi.instruments[0].name.lower().strip() != 'melody':
    removed_cnt += 1
    continue
  for idx, inst in enumerate(pretty_midi.instruments):
    inst.program = idx
  pretty_midi = clean_pretty_midi(pretty_midi)
  pretty_midi.write(f"{output_dir}\\{file_name}.mid")

print(f"{removed_cnt} songs removed")

