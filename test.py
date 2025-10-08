from train import *

def ngram_to_sentence(song, pizza=1):
  """Convert a sequence with ngrams to a sentence"""
  _, idx2tok = get_dictionaries()
  sentence = []
  for ngram in song:
    sentence += list(idx2tok[ngram.item()])
  return sentence # Now a sentence

# def best_

# def beam_search(model, song, pred_len, branch_factor):
#   current_nodes = []

def test_model(pred_len = 1, temp = 1, path='music_gen.py'):
  tok2idx, _ = get_dictionaries()
  sos_tok = tok2idx[('^',)]
  eos_tok = tok2idx[('$',)]
  model, _ = load_model(10000, path=path) # TODO: fix
  max_len = 40

  song = torch.ones((1,max_len+1), dtype=torch.long) * sos_tok

  for i in range(max_len//pred_len):
    # batch_size is first for model.predict
    prediction, is_done = model.predict(song, pred_len, temp) # (batch_size, output_len), bool
    song = torch.cat([song, prediction], dim=1)
    if is_done: break

  song = song[0] # Get first batch

  # DECODE
  min_note = 1*12
  max_note = 8*12
  fs = 100
  sentence = ngram_to_sentence(song)
  # print(sentence)
  piano_roll = sentence_to_piano_roll(sentence, min_note, max_note)
  # print(piano_roll)

  directory_path = 'pizzas'
  file_count = len([entry for entry in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, entry))])

  export_piano_roll(piano_roll, directory_path, 'beautiful_pizza_'+str(file_count), min_note, max_note, fs)
  show_midi(piano_roll)

test_model(1, 0.5)
