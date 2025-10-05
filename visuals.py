import matplotlib.pyplot as plt

def show_midi(piano_roll):
  fig, ax = plt.subplots()
  plt.imshow(piano_roll)
  plt.tight_layout()
  ax.invert_yaxis()
  plt.show()

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

