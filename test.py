from train import *


# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def ngram_to_sentence(song):
  """Convert a sequence with ngrams to a sentence"""
  _, idx2tok = get_dictionaries()
  sentence = []
  for ngram in song:
    sentence += list(idx2tok[ngram.item()])
  return sentence # Now a sentence

def test_model(max_len=1500, temp=0.5, path='music_gen.pt'):
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  tok2idx, idx2tok = get_dictionaries()
  sos_tok = tok2idx[('^',)]
  eos_tok = tok2idx[('$',)]
  model, _ = load_model(path=path, device=device) # TODO: fix

  song = torch.ones((1, 1), dtype=torch.long) * sos_tok

  with torch.no_grad():
    prediction = model.predict(song, max_len, temp)
  song = torch.cat([song, prediction], dim=1)
  
  # for i in range(max_len):
  #   # batch_size is first for model.predict
  #   # prediction, is_done = model.predict(song, pred_len, temp) # (batch_size, output_len), bool
  #   prediction = model.predict(song, pred_len, temp) # (batch_size, output_len)
  #   # print(song.shape, prediction.shape)
  #   song = torch.cat([song, prediction], dim=1)
  #   # if is_done: break

  song = song[0] # Get first batch
  for tok in song:
    print(idx2tok[tok.item()], end= ' ')
  # DECODE
  min_note = 1*12
  max_note = 8*12
  fs = 100
  sentence = ngram_to_sentence(song)
  # print(sentence)
  piano_roll = sentence_to_piano_roll(sentence, min_note, max_note)
  # print(piano_roll)

  directory_path = cwd / 'songs'
  file_count = len([entry for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])

  export_piano_roll(piano_roll, directory_path, 'song_'+str(file_count), min_note, max_note, fs)
  show_midi(piano_roll)

test_model()