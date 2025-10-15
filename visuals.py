import matplotlib.pyplot as plt
import pretty_midi as pm

def show_midi(piano_roll, aspect='equal'):
  fig, ax = plt.subplots()
  ax.imshow(piano_roll, aspect=aspect, interpolation='none')     
  ax.invert_yaxis()
  plt.tight_layout()
  plt.show()

def show_pretty_midi(pretty_midi: pm.PrettyMIDI, aspect='equal'):
  piano_roll = pretty_midi.get_piano_roll(fs=100)
  show_midi(piano_roll, aspect=aspect)

def compare_pretty_midi(pretty_midi_0: pm.PrettyMIDI, pretty_midi_1: pm.PrettyMIDI, start_sec=0, end_sec=None):
  piano_roll_0 = pretty_midi_0.get_piano_roll(fs=100)
  piano_roll_1 = pretty_midi_1.get_piano_roll(fs=100)
  compare_midis(piano_roll_0, piano_roll_1, start_sec, end_sec)

def compare_midis(piano_roll_0, piano_roll_1, start_sec=0, end_sec=None):
  if end_sec is None:
    end_sec = piano_roll_0.shape[1] / 100  # Assuming fs=100

  piano_roll_0 = piano_roll_0[:, int(start_sec*100):int(end_sec*100)]
  piano_roll_1 = piano_roll_1[:, int(start_sec*100):int(end_sec*100)]

  fig, axs = plt.subplots(
    2, 1, 
    sharey=True, 
    figsize=(8, 10),            # make taller
    gridspec_kw={'hspace': 0.1} # reduce vertical spacing
  )

  axs[0].imshow(piano_roll_0, aspect='auto')
  axs[0].invert_yaxis()
  axs[1].imshow(piano_roll_1, aspect='auto')
  axs[1].invert_yaxis()
  # axs[0].set_title('Original MIDI')
  # axs[1].set_title('Cleaned MIDI')
  for ax in axs:
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Note (semitone)')

  plt.tight_layout()
  plt.show()

