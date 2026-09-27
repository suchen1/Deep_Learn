import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time

def build_vocab():
    unique_word,all_word = [],[]
    for line in open('./data/jaychou_lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line)
        # print(words)
        all_word.extend(words)
        for word in words:
            if word not in unique_word:
                unique_word.append(word)
        all_word.append(" ")
    # print(f"unique_word.count = {len(unique_word)}") # 5703
    word_to_index = {word: index for index, word in enumerate(unique_word)}
    corpus_index = [word_to_index[word] for word in all_word]
    #print(word_to_index)
    #print(corpus_index)
    return word_to_index, corpus_index, unique_word, len(unique_word)

class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_index, num_chars):
        self.corpus_index = corpus_index
        self.num_chars = num_chars
        self.word_count = len(corpus_index)
        self.number = self.word_count - self.num_chars

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        start = min(max(idx,0), self.word_count - self.num_chars - 1)
        return (torch.tensor(self.corpus_index[start:start + self.num_chars]),
                torch.tensor(self.corpus_index[start + 1:start + self.num_chars + 1]))

class LyricsModel(nn.Module):
    def __init__(self,vocab_size):
        super.__init__()
        self.ebd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256)
        self.rnn = nn.RNN(256, 256, num_layers=1,batch_first=True)
        self.out = nn.Linear(256, vocab_size)

    def forward(self,input, h0):
        input = self.ebd(input)
        output, h1 = self.rnn(input, h0)
        output = self.out(output)
        return output, h1


if __name__ == '__main__':
    word_to_index, corpus_index, unique_word, vocab_size = build_vocab()
    print(f"vocab_size = {vocab_size}")
    print(f"word_to_index = {word_to_index}")
    print(f"corpus_index = {corpus_index[:10]}")  # 打印前10个词的索引
    lyricsdataset = LyricsDataset(corpus_index, 5)
    print(len(lyricsdataset))
    print(lyricsdataset[0])
