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
  enc_input_size=16,
  dec_input_size=16,
  hidden_size=64,
  num_layers=2,
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
  epochs=10,
  batch_size=16,
  lr=0.001,
  forcing=0.5,
  vocab_size=10000,
  target_len=10,
  model_path='music_gen.pt',
  save_freq=10,
  data_path='MusicRNN\\cleaned_midi'
):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dataset = Corpus(data_path, vocab_size)
  
  loader = DataLoader(dataset, batch_size, True, collate_fn=collate_fn)

  # Load the model
  model, epoch = load_model(vocab_size, path=model_path)
  model = model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optim = torch.optim.Adam(model.parameters(), lr=lr)

  for epoch in range(epoch, epochs):
    progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", position=0, nrows=2, leave=False)
    avg_loss = 0
    # num_losses = 0
    for batch_idx, batch in enumerate(progress_bar):
      batch = batch.to(device)
      seq_lens = range(1, (batch.shape[1]), target_len)
      progress_bar2 = tqdm(seq_lens, desc=f"Batch {batch_idx + 1}/{len(progress_bar)}", position=1, leave=False)
      kf = 0
      for seq_len in progress_bar2: # (batch_size, seq_len)
        if hasattr(model, 'reset_states'):
          model.reset_states()
        input = batch[:,:seq_len].detach() # (batch_size, seq_len)
        target = batch[:,seq_len:min(seq_len+target_len, batch.shape[1])].detach() # (batch_size, target_length)
        # kf += 1
        # if kf < 3:
        #   print(input, target)
        logits = model(input, target, forcing) # (batch_size, target_length, vocab_size)
        # CrossEntropyLoss applies softmax
        loss = loss_fn(logits.reshape(-1, vocab_size), target.reshape(-1)) # (1,)
        # num_losses += 1
        avg_loss += loss.item()
        # print(loss)
        loss.backward()
        # clip grads
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
        # del logits, loss
        # torch.cuda.empty_cache()

      if (batch_idx + 1) % save_freq == 0:
        print(f"Saving model... avg loss={(avg_loss/len(seq_lens)):0.4f}")
        save_model(model, epoch, model_path)
        avg_loss = 0
        # num_losses = 0
if __name__ == '__main__':
  train(target_len=1, forcing=1.0)