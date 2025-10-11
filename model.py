import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicGen(nn.Module):
  def __init__(self, vocab_size, input_size, hidden_size, num_layers, sos_tok, eos_tok):
    super().__init__()
    
    self.sos_tok = sos_tok
    self.eos_tok = eos_tok
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    
    # Combine context vector with output from decoder
    # A context vector is derived from taking the dot product
    # of the output of the encoder with the output of the decoder
    self.combine_ctx = nn.Linear(2 * hidden_size, hidden_size)
    # Map input dimension to the size of the input for the encoder
    self.embed = nn.Embedding(vocab_size, input_size)

    # The size of the output from the decoder is mapped to the vocab size
    self.hidden2vocab = nn.Linear(hidden_size, vocab_size)

    # A GRU is a type of RNN with gates that help the model learn complex patterns
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

  # outputs logits
  def attention(self, outputs):
    """
    attention mechanism

    Parameters
    ----------
    - outputs (pytorch.tensor) : (batch, seq_len, hidden_size)

    Returns:
    -------
    - logits (pytorch.tensor) : (batch_size, vocab_size)
    """
    predicted = outputs[:,-1] #(batch_size, hidden_size)
    previous = outputs[:, :-1] # (batch_size, seq_length, hidden_size)
    similarity_scores = torch.bmm(previous, predicted[..., None]) #(batch_size, seq_length, 1)
    similarity_scores = similarity_scores.squeeze(2) # (batch_size, seq_len)
    similarity_scores = F.softmax(similarity_scores, dim=1)

    # weighting the previous hn's by their outputs similarity to predicted output
    co = torch.bmm(similarity_scores[:, None, :], previous) # (batch_size, 1, hidden_size)
    co = co.squeeze(1) # (batch_size, hidden_size)

    combined = torch.cat((predicted, co), dim=1) # (batch_size, 2 * hidden_size)
    combined = self.combine_ctx(combined)
    combined = torch.tanh(combined)
    logits = self.hidden2vocab(combined) # (batch_size, vocab_size)
    
    return logits
  
  @staticmethod
  def teacher_forcing(logits, ground_truth, forcing):
    if random.random() < forcing:
      result = ground_truth
      result = result[:, None] # (batch_size, 1)
    else:
      result = torch.argmax(logits, dim=1, keepdim=True) # (batch_size, 1)
    return result # (batch_size, 1)
  
  @staticmethod
  def temperature(logits, temp):
    if temp == 0:
      return torch.argmax(logits, dim=1, keepdim=True) # (batch_size, 1)
    
    probs = F.softmax(logits / temp, 1) # (batch_size, vocab_size)
    next_tok = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
    return next_tok

  def forward(self, input, targets, forcing=0.5):
    device = targets.device
    batch_size, target_length = targets.shape

    emb = self.embed(input) # (batch_size, seq_len, input_size)
    # output: (batch_size, seq_len, hidden_size)
    # hn: (num_layers, batch_size, hidden_size)
    outputs, hn = self.gru(emb)
    logits = self.attention(outputs) # logits: (batch_size, vocab_size)
    input = self.teacher_forcing(logits, targets[:,0], forcing) # (batch_size, 1)

    logit_list = [logits]
    for seq_idx in range(1, target_length):
      # print(input)
      emb = self.embed(input) # (batch_size, 1, input_size)
      # output: (batch_size, 1, hidden_size)
      # hn: (num_layers, batch_size, hidden_size)
      output, hn = self.gru(emb, hn)
      outputs = torch.cat((outputs, output), dim=1) # (batch_size, seq_len, hidden_size)

      logits = self.attention(outputs) # logits: (batch_size, vocab_size)
      logit_list += [logits]
      input = self.teacher_forcing(logits, targets[:, seq_idx], forcing) # (batch_size, 1)

    return torch.stack(logit_list, dim=1) # (batch_size, target_length, vocab_size)
  
  def predict(self, input, max_len, temp):
    self.eval()
    emb = self.embed(input) # (batch_size, seq_len, input_size)
    # output: (batch_size, seq_len, hidden_size)
    # hn: (num_layers, batch_size, hidden_size)
    outputs, hn = self.gru(emb)
    logits = self.attention(outputs) # logits: (batch_size, vocab_size)
    torch.set_printoptions(threshold=float('inf'))
    # print(logits[0, :100])
    input = self.temperature(logits, temp) # (batch_size, 1)
    new_toks = [input.squeeze(1)]
    for seq_idx in range(1, max_len): # TODO stop if end of seq char
      emb = self.embed(input) # (batch_size, 1, input_size)
      # output: (batch_size, 1, hidden_size)
      # hn: (num_layers, batch_size, hidden_size)
      output, hn = self.gru(emb, hn)
      outputs = torch.cat((outputs, output), dim=1) # (batch_size, seq_len, hidden_size)

      logits = self.attention(outputs) # logits: (batch_size, vocab_size)
      input = self.temperature(logits, temp) # (batch_size, 1)
      new_toks += [input.squeeze(1)]

    return torch.stack(new_toks, dim=1) # return: (batch_size, output_len)