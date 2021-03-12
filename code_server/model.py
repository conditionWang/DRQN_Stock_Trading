import torch
import torch.nn as nn


class DQN(nn.Module):
	def __init__(self, state_size, action_size):
		super(DQN, self).__init__()
		self.first_two_layers = nn.Sequential(
			nn.Linear(state_size, 256),
			nn.ELU(),
			nn.Linear(256, 256),
			nn.ELU()
		)
		self.lstm = nn.LSTM(256, 256, 1, batch_first=True)
		self.last_linear = nn.Linear(256, 3)

# Data Flow Protocol:
# 1. network input shape: (batch_size, seq_length, num_features)
# 2. LSTM output shape: (batch_size, seq_length, hidden_size)
# 3. Linear input shape:  (batch_size * seq_length, hidden_size)
# 4. Linear output: (batch_size * seq_length, out_size)

	def forward(self, input):
		# rint(input.size())
		x = self.first_two_layers(input)
		# print(x.size())
		
		lstm_out, hs = self.lstm(x)
		# print(lstm_out.size())

		batch_size, seq_len, mid_dim = lstm_out.shape
		linear_in = lstm_out.contiguous().view(seq_len * batch_size, mid_dim)
		# linear_in = lstm_out.contiguous().view(-1, lstm_out.size(2))

		# linear_in = lstm_out.reshape(-1, hidden_size) 
		return self.last_linear(linear_in)