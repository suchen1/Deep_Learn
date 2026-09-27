import torch
import torch.nn as nn
import jieba

test = "我家大门常打开，欢迎朋友常来玩，北京欢迎你"

word = jieba.lcut(test)
embd = nn.Embedding(num_embeddings=len(word), embedding_dim=4)
for i, word in enumerate(word):
    print(f"第{i}个词是：{word}")
    word2vec = embd(torch.tensor(i))
    print(f"词向量是：{word2vec}")