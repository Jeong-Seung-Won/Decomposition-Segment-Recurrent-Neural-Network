class TTMultiheadAttention(nn.Module):
    def __init__(self, input_dim, rank, num_heads, dropout_rate=0.1):
        super(TTMultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        tt_linear = TTLinear(input_dim, rank)
        self.ar, self.br, self.cr = tt_linear.forward()
        self.fc_out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.input_dim = input_dim
        self.rank = rank[2]

    def split_heads(self, x, y):
        x = x.view(x.shape[0], y, -1, x.shape[2], x.shape[3])
          
        return x

    def forward(self, query, key, value):
        df = torch.stack([query, key, value])
        size = df.shape
        df = self.split_heads(df, size[1] // self.rank)

        ar = self.ar
        br = self.br
        cr = self.cr

        attention = torch.matmul(df.permute(3, 1, 2, 4, 0), ar)
        attention = torch.matmul(attention.permute(0, 1, 4, 3, 2), br.permute(0, 2, 1))
        attention = attention / torch.sqrt(torch.tensor(self.head_dim, dtype = torch.float32))
        attention = F.softmax(attention, dim = -1)
        out = torch.matmul(attention, cr.permute(2, 1, 0)).to(device)
        out = out.permute(2, 1, 4, 0, 3)
        out = out.reshape(size)
        out = out.sum(dim = 0).view(query.shape)
        
            
        return out
