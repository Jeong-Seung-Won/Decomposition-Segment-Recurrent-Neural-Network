class DASR(nn.Module):
    def __init__(self, batch_size, input_dim, enc_in, rank, seq_len, pred_len, patch_len, nhead, dropout, hidden_dim):
        super(DASR, self).__init__()

        self.lucky = nn.Embedding(enc_in, hidden_dim // 2)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.hidden_dim = hidden_dim
        
        self.linear_patch = nn.Linear(self.patch_len, self.hidden_dim)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bias=True,
            batch_first=True,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.pred_len // self.patch_len, self.hidden_dim // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.hidden_dim // 2))

        self.dropout = nn.Dropout(dropout)
        self.linear_patch_re = nn.Linear(self.hidden_dim, self.patch_len)

        self.self_attn1 = nn.MultiheadAttention(enc_in, nhead)
        self.self_attn2 = TTMultiheadAttention(input_dim, rank, input_dim // 2, dropout)

    def forward(self, x):
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last # (batch_size, seq_len, enc_in)

        attn = self.self_attn1(x, x, x)[0]
        x = x + attn

        B, L, C = x.shape
        N = self.seq_len // self.patch_len
        M = self.pred_len // self.patch_len
        W = self.patch_len
        d = self.hidden_dim

        xw = x.permute(0, 2, 1).reshape(B * C, N, -1)  # B, L, C -> B, C, L -> B * C, N, W
        xd = self.linear_patch(xw)  # B * C, N, W -> B * C, N, d
        enc_in = self.relu(xd)
        enc_in = self.self_attn2(enc_in, enc_in, enc_in)

        enc_out = self.gru(enc_in)[1].repeat(1, 1, M).view(1, -1, self.hidden_dim)  # 1, B * C, d -> 1, B * C, M * d -> 1, B * C * M, d

        dec_in = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(B*C, 1, 1),  # M, d // 2 -> 1, M, d // 2 -> B * C, M, d // 2
            self.channel_emb.unsqueeze(1).repeat(B, M, 1)  # C, d // 2 -> C, 1, d // 2 -> B * C, M, d // 2
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # B * C, M, d -> B * C * M, d -> B * C * M, 1, d

        dec_in = self.self_attn2(dec_in, dec_in, dec_in)
        dec_out = self.gru(dec_in, enc_out)[0]  # B * C * M, 1, d

        yd = self.dropout(dec_out)
        yw = self.linear_patch_re(yd)  # B * C * M, 1, d -> B * C * M, 1, W
        y = yw.reshape(B, C, -1).permute(0, 2, 1)  # B, C, H

        y = y + seq_last
        return y