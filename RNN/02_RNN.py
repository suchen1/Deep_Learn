import torch
import torch.nn as nn

rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1) # (词向量维度，隐藏层维度，隐藏层层数)

input = torch.randn(size=(5,32, 128)) # (句子的次数量，句子个数，词向量维度)
h0 = torch.randn(size=(1, 32, 256)) # (隐藏层层数，句子个数，隐藏层维度)

output, h1 = rnn(input, h0)

print(f"output.shape = {output.shape}") # (5, 32, 256) 5个词，32个句子，256维的隐藏层
print(f"h1.shape = {h1.shape}") # (1, 32, 256) 1层隐藏层，32个句子，256维的隐藏层

