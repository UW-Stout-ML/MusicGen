import pretty_midi as pm

template = pm.PrettyMIDI('POP9092_cleaned/001-v1.mid')
notes = template.instruments[0].notes

for i in range(128):
  song = pm.PrettyMIDI()
  inst = pm.Instrument(program=i)
  min_time = notes[0].start

  for note in notes:
    inst.notes.append(pm.Note(velocity=note.velocity, pitch=note.pitch, start=note.start - min_time, end=note.end - min_time))

  song.instruments.append(inst)
  song.write(f"tests/test_{i}.mid")