import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class MusicEncoder(nn.Module):
  def __init__(self, vocab_size, input_size, hidden_size, num_layers):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, input_size)
    self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

  def forward(self, x):
    # input: (batch_size, 1)
    # output: (batch_size, seq_len, 2 * hidden_size)
    # hn: (2 * num_layers, batch_size, hidden_size)
    input = self.embed(x)
    output, hn = self.gru(input)
    return output, hn

class MusicDecoder(nn.Module):
  def __init__(self, vocab_size, input_size, hidden_size, num_layers, sos_tok, eos_tok):
    super().__init__()
    self.sos_tok = sos_tok
    self.eos_tok = eos_tok
    self.proj_h = nn.Linear(2 * hidden_size, hidden_size)
    self.proj_o = nn.Linear(2 * hidden_size, hidden_size)
    self.combine_ctx = nn.Linear(3 * hidden_size, hidden_size)
    self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
    self.embed = nn.Embedding(vocab_size, input_size)
    self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
    self.num_layers = num_layers
    self.hidden_size = hidden_size

  def forward(self, enc_out, enc_hn, targets, forcing=0.5):
    device = targets.device
    batch_size, target_length = targets.shape
    dec_inp = torch.full((batch_size, 1), self.sos_tok, dtype=torch.long, device=device)
    enc_hn = enc_hn.view(self.num_layers, 2, batch_size, self.hidden_size)
    dec_h_neg_1 = torch.cat((enc_hn[:, 0, ...], enc_hn[:, 1, ...]), dim=2) # (num_layers, batch_size, 2 * hidden_size) 
    proj_h_neg_1 = self.proj_h(dec_h_neg_1) # (num_layers, batch_size, hidden_size)

    # Attention!
    proj_enc_out = self.proj_o(enc_out) # (batch_size, seq_len, hidden_size)
    outputs = []
    dec_hn = proj_h_neg_1
    for seq_idx in range(target_length):
      emb = self.embed(dec_inp) # (batch_size, input_size)
      # output: (batch_size, 1, hidden_size)
      # dec_hn: (num_layers, batch_size, hidden_size)
      dec_output, dec_hn = self.gru(emb, dec_hn)
      # dec_hn = dec_hn.detach()
      # output: (batch_size, hidden_size)
      dec_output = dec_output.squeeze()
      similarity_scores = torch.bmm(proj_enc_out, dec_output[..., None]) # (batch_size, seq_len, 1)
      similarity_scores = similarity_scores.squeeze(2) # (batch_size, seq_len)
      similarity_scores = F.softmax(similarity_scores, dim=1)
      co = torch.bmm(similarity_scores[:, None, :], enc_out) # (batch_size, 1, 2 * hidden_size)
      co = co.squeeze(1) # (batch_size, 2 * hidden_size)
      combined = torch.cat((dec_output, co), dim=1) # (batch_size, 3 * hidden_size)
      combined = self.combine_ctx(combined)
      combined = torch.tanh(combined)
      logits = self.hidden2vocab(combined) # (batch_size, vocab_size)
      outputs += [logits]
      if random.random() < forcing:
        dec_inp = targets[:, seq_idx]
        dec_inp = dec_inp[:, None] # (batch_size, 1)
      else:
        dec_inp = torch.argmax(logits, dim=1, keepdim=True) # (batch_size, 1)

    return torch.stack(outputs, dim=1) # (batch_size, target_length, vocab_size)
  
  def inference(self, enc_out, enc_hn, max_len, temp):
    device = enc_out.device
    batch_size = 1
    dec_inp = torch.full((batch_size, 1), self.sos_tok, dtype=torch.long, device=device)
    enc_hn = enc_hn.view(self.num_layers, 2, batch_size, self.hidden_size)
    dec_h_neg_1 = torch.cat((enc_hn[:, 0, ...], enc_hn[:, 1, ...]), dim=2) # (num_layers, batch_size, 2 * hidden_size) 
    proj_h_neg_1 = self.proj_h(dec_h_neg_1) # (num_layers, batch_size, hidden_size)

    # Attention!
    proj_enc_out = self.proj_o(enc_out) # (batch_size, seq_len, hidden_size)
    outputs = []
    # finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)
    is_done = False

    for _ in range(max_len):
      emb = self.embed(dec_inp) # (batch_size, input_size)
      # dec_output: (batch_size, 1, hidden_size)
      # dec_hn: (num_layers, batch_size, hidden_size)
      # print(emb.shape, proj_h_neg_1.shape)
      dec_output, dec_hn = self.gru(emb, proj_h_neg_1)
      # dec_output: (batch_size, hidden_size)
      dec_output = dec_output.squeeze(1)
      # proj_enc_out: (batch_size, seq_len, hidden_size)
      # print(proj_enc_out.shape, dec_output[..., None].shape)
      similarity_scores = torch.bmm(proj_enc_out, dec_output[..., None]) # (batch_size, seq_len, 1)
      similarity_scores = similarity_scores.squeeze(2) # (batch_size, seq_len)
      similarity_scores = F.softmax(similarity_scores)
      co = torch.bmm(similarity_scores[:, None, :], enc_out) # (batch_size, 1, 2 * hidden_size)
      co = co.squeeze(1) # (batch_size, 2 * hidden_size)
      combined = torch.cat((dec_output, co), dim=1) # (batch_size, 3 * hidden_size)
      combined = self.combine_ctx(combined)
      combined = torch.tanh(combined)
      logits = self.hidden2vocab(combined) # (batch_size, vocab_size)
      probs = F.softmax(logits / temp, 1) # (batch_size, vocab_size)
      
      # Pick the pizza!
      next_tok = torch.multinomial(probs, num_samples=1)[:, 0] # (batch_size,)
      outputs += [next_tok] # (batch_size,)
      next_tok = next_tok[:, None] # (batch_size, 1)
      dec_inp = next_tok

      # finished = finished | (next_tok == self.eos_tok)
      if next_tok.item() == self.eos_tok:
        is_done = True
        break
    
    # outputs: (output_len, batch_size)
    return torch.stack(outputs, dim=1), is_done # (batch_size, output_len), bool

class Seq2Seq(nn.Module):
  def __init__(self, vocab_size, enc_input_size, dec_input_size, hidden_size, num_layers, sos_tok, eos_tok):
    super().__init__()
    self.enc = MusicEncoder(vocab_size, enc_input_size, hidden_size, num_layers)
    self.dec = MusicDecoder(vocab_size, dec_input_size, hidden_size, num_layers, sos_tok, eos_tok)

  def forward(self, input, targets, forcing):
    enc_out, enc_hn = self.enc(input)
    return self.dec(enc_out, enc_hn, targets, forcing)

  def predict(self, input_tensor, max_len, temp):
    self.eval()

    with torch.no_grad():
      enc_out, enc_hn = self.enc(input_tensor)
      return self.dec.inference(enc_out, enc_hn, max_len, temp)
