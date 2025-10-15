from train import *


def test_model(max_len=1000, temp=0.9, path='music_gen.pt'):
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tok2idx, idx2tok = get_dictionaries()
  sos_tok = tok2idx[('^',)]
  eos_tok = tok2idx[('$',)]
  model, _ = load_model(path=path, device=device) # TODO: fix

  song = torch.ones((1, 1), dtype=torch.long) * sos_tok

  with torch.no_grad():
    prediction = model.predict(song, max_len, temp)
  song = torch.cat([song, prediction], dim=1)

  song = song[0] # Get first batch
  fs = 100
  sentence = ngram_to_sentence(song, get_dictionaries())

  print(sentence)
  
  directory_path = cwd / 'output_songs'
  file_count = len([entry for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])
  file_path = os.path.join(directory_path, f"song_{str(file_count)}.mid")

  pretty_midi = sentence_to_pretty_midi(sentence, fs)
  
  show_pretty_midi(pretty_midi)
  
  if not os.path.exists(directory_path):
    os.makedirs(directory_path)

  pretty_midi.write(file_path)

test_model()