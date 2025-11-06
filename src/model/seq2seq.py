import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size=32000, d_model=256, nhead=4, num_layers=4, dim_ff=768, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.lm = nn.Linear(d_model, vocab_size)
        
    def forward(self, src_ids, target_ids):
        B, S = src_ids.size()
        _, T = target_ids.size()
        pos_s = torch.arange(S, device=src_ids.device).unsqueeze(0)
        pos_t = torch.arange(T, device=target_ids.device).unsqueeze(0)
        src = self.emb(src_ids) + self.pos(pos_s)
        tgt = self.emb(target_ids) + self.pos(pos_t)
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        logits = self.lm(out)
        return logits