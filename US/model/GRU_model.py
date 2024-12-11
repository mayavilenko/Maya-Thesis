
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.use_deterministic_algorithms(True)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, seed=42):
        super(GRUModel, self).__init__()
        torch.manual_seed(seed)  # Ensure reproducibility

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())
        
        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.context_vector = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.xavier_uniform_(self.attention_weights) #xavier_uniform_ is a method for initializing the weights of neural network layers in a way that helps to maintain the variance of the weights across layers, which can help in stabilizing the training process.
        nn.init.xavier_uniform_(self.context_vector)

    def forward(self, gru_output):
        # Compute attention scores
        score = torch.tanh(torch.matmul(gru_output, self.attention_weights))
        attention_weights = torch.matmul(score, self.context_vector).squeeze(-1) #squeeze(-1) is used to remove the last dimension from the attention_weights tensor, which is of size 1 after the matrix multiplication with self.context_vector.
        attention_weights = F.softmax(attention_weights, dim=1)

        # Compute the context vector as the weighted sum of GRU outputs
        context_vector = torch.sum(gru_output * attention_weights.unsqueeze(-1), dim=1)
        return context_vector, attention_weights

class GRUWithAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, seed=42):
        super(GRUWithAttentionModel, self).__init__()
        torch.manual_seed(seed)  # Ensure reproducibility

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)

        # Attention layer
        self.attention = Attention(hidden_dim)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        gru_output, _ = self.gru(x, h0.detach())

        # Apply attention mechanism
        context_vector, attention_weights = self.attention(gru_output)

        # Convert the context vector to our desired output shape (batch_size, output_dim)
        out = self.fc(context_vector)

        return out, attention_weights
