from model import *
from data_loader import *

def save_model(model: Seq2Seq, epochs, path='music_gen.pt'):
  """Save the Seq2Seq model

  Parameters
  ----------
  model : Seq2Seq
    The model to save
  path : string
    Path to load the model

  """
  torch.save({
    'model': model.state_dict(),
    'epochs': epochs
  }, path)

def load_model(
  vocab_size,
  enc_input_size=64,
  dec_input_size=64,
  hidden_size=512,
  num_layers=3,
  path='music_gen.pt'
) -> tuple[Seq2Seq, int]:
  """Load the Seq2Seq model

  Parameters
  ----------
  path : string
    Path to the saved model

  Returns
  -------
  The model and the number of epochs.
  """
  tok2idx, _ = get_dictionaries()
  sos_tok = tok2idx[('^',)]
  eos_tok = tok2idx[('$',)]
  model = Seq2Seq(vocab_size, enc_input_size, dec_input_size, hidden_size, num_layers, sos_tok, eos_tok)
  
  try:
    dic = torch.load(path)
    model.load_state_dict(dic['model'])
    return model, dic['epochs']
  except: # No model file found
    return model, 0

def train(
  epochs=10000,
  batch_size=32,
  lr=0.001,
  forcing=0.5,
  vocab_size=10000,
  target_len=20,
  model_path='music_gen.pt',
  save_freq=2,
  data_path='midis',
  max_songs=None,
  min_song_len=30
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = Corpus(data_path, vocab_size, max_songs=max_songs, min_song_len=min_song_len, rand=True, target_len=target_len, cap=256)
  
  if len(dataset) == 0:
    print("Error: The dataset is empty!")
    exit(1)

  loader = DataLoader(dataset, batch_size, True, collate_fn=get_collate_fn(target_len))

  # Load the model
  model, epoch = load_model(vocab_size, path=model_path)
  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in tqdm(range(epoch, epochs), 'Training...'):
    total_loss = 0
    total_tok = 0
    # progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, nrows=2, leave=False)
    # for batch_idx, x in enumerate(progress_bar):
    for x in loader:
      input = x[0]
      target = x[1]
      input = input.to(device)
      target = target.to(device)

      # print(input, target)

      if hasattr(model, 'reset_states'):
        model.reset_states()
      logits = model(input, target, forcing) # (batch_size, target_length, vocab_size)
      # CrossEntropyLoss applies softmax
      loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1)) # (1,)
      total_loss += loss.item() * target.numel()
      total_tok += target.numel() # Gives you the number of batches
      optim.zero_grad()
      loss.backward()
      # clip grads
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optim.step()
    
    if (epoch + 1) % save_freq == 0:
      print(f"Saving model... avg loss={(total_loss/total_tok):0.4f}")
      save_model(model, epoch, model_path)

if __name__ == '__main__':
  train()
