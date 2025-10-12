from model import *
from data_loader import *
from pathlib import Path

cwd = Path(__file__).parent
data_dir = cwd / 'cleaned_midi'

def save_model(model: MusicGen, epochs, path='music_gen.pt'):
  """Save the MusicGen model

  Parameters
  ----------
  model : MusicGen
    The model to save
  path : string
    Path to load the model

  """
  torch.save({
    'model': model.state_dict(),
    'epochs': epochs,
    'vocab_size': model.vocab_size
  }, path)

def load_model(
  input_size=32,
  hidden_size=256,
  num_layers=2,
  path='music_gen.pt'
) -> tuple[MusicGen, int]:
  """Load the MusicGen model

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
  vocab_size = len(tok2idx)
  model = MusicGen(vocab_size, input_size, hidden_size, num_layers, sos_tok, eos_tok)
  
  try:
    dic = torch.load(path)
    model.load_state_dict(dic['model'])
    return model, dic['epochs']
  except: # No model file found
    print("Model not found, creating a new one...")
    return model, 0

def train(
  # Training parameters
  epochs=10000000,
  batch_size=32,
  lr=0.0001,
  forcing=0.5,
  input_len=44,
  target_len=20,
  save_freq=20,

  # Dataset parameters
  max_vocab_size=10000,
  data_path=data_dir,
  max_songs=10000,
  max_corp_songs=100,
  model_path='music_gen.pt',
):
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = Corpus(
    data_path,
    input_len,
    target_len,
    max_corp_songs,
    max_songs,
    max_vocab_size
  )
  
  vocab_size = dataset.vocab_size

  print(f"Vocab size: {vocab_size}")

  if len(dataset) == 0:
    print("Error: The dataset is empty!")
    exit(1)

  loader = DataLoader(dataset, batch_size, True)

  # Load the model
  model, epoch = load_model(path=model_path)
  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in tqdm(range(epoch, epochs), 'Training...'):
    total_loss = 0
    total_tok = 0
    for x in loader:
      input = x[0]
      target = x[1]
      input = input.to(device)
      target = target.to(device)

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