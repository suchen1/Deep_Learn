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
        super().__init__()
        self.ebd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=256)
        self.rnn = nn.RNN(256, 256, num_layers=1,batch_first=True)
        self.out = nn.Linear(256, vocab_size)

    def forward(self,input, h0):
        input = self.ebd(input)
        output, h1 = self.rnn(input, h0)
        output = self.out(output)
        return output, h1

    def init_hidden(self, bs):
        return torch.zeros(1, bs, self.rnn.hidden_size,
                           device=self.ebd.weight.device)  # 跟着模型走


def train(device):
    batch_size = 32
    word_to_index, corpus_index, unique_word, vocab_size = build_vocab()
    lyricsdataset = LyricsDataset(corpus_index, 16)
    model = LyricsModel(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataLoader = DataLoader(lyricsdataset, batch_size=batch_size, shuffle=True)
    epochs = 50
    for epoch in range(epochs):
        total_loss, total_samples, start = 0, 0, time.time()
        for x, y in dataLoader:
            x, y = x.to(device), y.to(device)           # 数据搬过去
            hidden = model.init_hidden(x.size(0))  # 初始化隐藏状态
            y_pred, hidden = model(x, hidden)
            loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total_samples:.4f}, Time: {time.time() - start:.2f}s")
    torch.save(model.state_dict(), './model/lyrics.pth')

def evaluate(startword,sentence_len,device):
    word_to_index, corpus_index, unique_word, vocab_size = build_vocab()
    model = LyricsModel(vocab_size).to(device)
    model.load_state_dict(torch.load('./model/lyrics.pth', map_location=device))
    model.eval()
    # input = torch.tensor([word_to_index[startword]], device=device).unsqueeze(0)  # (1,1)
    input = word_to_index[startword]
    hidden = model.init_hidden(1)
    generated_words = [startword]
    with torch.no_grad():
        for _ in range(sentence_len):
            output, hidden = model(torch.tensor([[input]],device=device), hidden)
            # next_word_index = torch.argmax(output[:, -1, :], dim=-1).item()
            next_word_index = torch.argmax(output)
            next_word = unique_word[next_word_index]
            generated_words.append(next_word)
            input = torch.tensor([[next_word_index]], device=device)  # (1,1)

    for idx in generated_words:
        print(idx,end=" ")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train(device)
    evaluate("余", 150, device)
