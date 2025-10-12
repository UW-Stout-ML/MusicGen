from model import *
from data_loader import *
from pathlib import Path

cwd = Path(__file__).parent
data_dir = cwd / 'multi_note_songs'

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
  hidden_size=1028,
  num_layers=3,
  path='music_gen.pt',
  device='cuda'
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
    print(f"Loading model file: {path}")
    dic = torch.load(path, map_location=device)
  except Exception as e: # No model file found
    print(e)
    print("Model not found, creating a new one...")
    exit()
    return model, 0
  
  model.load_state_dict(dic['model'])
  return model, dic['epochs']

def train(
  # Training parameters
  epochs=10000000,
  batch_size=32,
  lr=0.0001,
  forcing=0.5,
  input_len=128,
  target_len=1,
  save_freq=2,
  loss_criteria=0.0,

  # Dataset parameters
  max_vocab_size=10000,
  data_path=data_dir,
  max_songs=10000,
  prob_sos=0.0,
  max_corp_songs=float('inf'),
  model_path='music_gen.pt',
):
  # Calculate min and max input/target length
  min_input_len = input_len if isinstance(input_len, (int, float)) else list(input_len)[0]
  max_input_len = input_len if isinstance(input_len, (int, float)) else list(input_len)[-1]
  min_target_len = target_len if isinstance(target_len, (int, float)) else list(target_len)[0]
  max_target_len = target_len if isinstance(target_len, (int, float)) else list(target_len)[-1]
  min_song_len = max_input_len + max_target_len

  stage_idx = 0
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = Corpus(
    data_path,
    min_song_len,
    max_corp_songs,
    max_songs,
    max_vocab_size
  )
  
  vocab_size = dataset.vocab_size

  print(f"Vocab size: {vocab_size}")

  if len(dataset) == 0:
    print("Error: The dataset is empty!")
    exit(1)

  loader = DataLoader(dataset, batch_size, True, collate_fn=collate_fn(
    prob_sos, min_input_len, max_input_len, min_target_len, max_target_len
  ))
  
  # Load the model
  model, epoch = load_model(path=model_path, device=device)
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

    avg_loss = total_loss / total_tok

    if (epoch + 1) % save_freq == 0:
      print(f"Saving model... avg loss={avg_loss:0.4f}")
      save_model(model, epoch, model_path)

    # Update stage
    stage = stages[stage_idx]
    if avg_loss < loss_criteria:
      break # Next stage

def stage_train(stages):
  for stg_idx, stage in enumerate(stages):
    print(f"Stage {stg_idx + 1}")
    train(**stage)

if __name__ == '__main__':
  stages = [
    {
      'loss_criteria': 2.5,
      'save_freq': 10,
      'forcing': 1.0,
      'input_len': 32,
      'target_len': 1,
      'prob_sos': 0,
    },
    {
      'loss_criteria': 1.6,
      'save_freq': 10,
      'forcing': 1.0,
      'input_len': 50,
      'target_len': 12,
      'prob_sos': 0.01,
    },
    {
      'loss_criteria': 1.1,
      'save_freq': 2,
      'forcing': 1.0,
      'input_len': 100,
      'target_len': 50,
      'prob_sos': 0.01,
    },
    {
      'loss_criteria': 0.0,
      'save_freq': 2,
      'forcing': 1.0,
      'input_len': (100, 500),
      'target_len': 30,
      'prob_sos': 0.01,
      'batch_size':32,
    },
  ]

  stage_train(stages)
