import pretty_midi as pm
import os
from tqdm import tqdm

program_map = {
  'grandeur 1': 1,
  '': 1,
  'melody': 0,
  'bridge': 1,
  'gggg': 1,
  'hoa âm': 1,
  'piano4': 1,
  'midi 04': 1,
  'copy of halion sonic se 01': 1,
  'piano': 1,
  'piano`': 1,
  'lót': 1,
  'meldoy': 0,
  'grandeur 2': 1,
  'halion sonic se 01': 1,
  'steinway grand piano': 1,
  'dem': 1,
  'melopy': 0,
  'cau': 1,
  'pianio': 1,
  'cau chen': 1,
  'instrument track 01': 1,
  'track 4': 1
}

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
  # pretty_midi.instruments = list(filter(lambda x: x.name in ('MELODY'), pretty_midi.instruments))
  for instrument in pretty_midi.instruments:
    for note in instrument.notes:
      top_vel = max(top_vel, note.velocity)
      min_time = min(min_time, note.start)
  for idx, instrument in enumerate(pretty_midi.instruments):
    # instrument.program = idx
    for note in instrument.notes:
      note.velocity = int(note.velocity / max_vel * max_vel)
      note.start -= min_time
  return pretty_midi

input_dir = 'POP9092'
output_dir = 'POP9092_cleaned'
midis = get_midis(input_dir)

if not os.path.exists(output_dir):
  os.mkdir(output_dir)

# Sanity check
cnt = 0
inst_names = set()
for idx, midi in enumerate(tqdm(midis, 'Cleaning midis')):
  pretty_midi = pm.PrettyMIDI(midi)
  if len(pretty_midi.instruments) == 0:
    cnt += 1
    continue
  for inst in pretty_midi.instruments:
    inst_name = inst.name.lower().strip()
    inst_names.add(inst_name)
    if inst_name not in program_map:
      inst.program = 1
    else:
      inst.program = program_map[inst_name]
  if pretty_midi.instruments[0].name.lower().strip() != 'melody':
    cnt += 1
    continue
  pretty_midi = clean_pretty_midi(pretty_midi)
  pretty_midi.write(f"{output_dir}\\song_{idx}.mid")
  # if idx > 10: break
print(f"{cnt} misalignments")
print('instruments:', inst_names)

