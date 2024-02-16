class TTLinear(nn.Module):
    def __init__(self, input_dim, rank):
        super(TTLinear, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        query_weight = self.query.weight
        key_weight = self.key.weight
        value_weight = self.value.weight
        weight_matrix = torch.stack([query_weight, key_weight, value_weight])

        weight_matrix_np = weight_matrix.detach().cpu().numpy()
        factors = matrix_product_state(weight_matrix.detach().numpy(), rank)

        self.factors = nn.ParameterList([nn.Parameter(torch.FloatTensor(factor)) for factor in factors])
        
    def forward(self):
        return self.factors[0], self.factors[1], self.factors[2]