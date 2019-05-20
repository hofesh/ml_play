import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import pdb
import time
import string

to_gpu = False
def gpu(m):
    if to_gpu:
        return m.cuda()
    return m

def ints_to_tensor(ints):
    return gpu(torch.tensor(ints).long().transpose(1, 0))

# build the model using the pytorch nn module
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, batch_size, embedding_dim):
        super(CharLSTM, self).__init__()
        
        # init the meta parameters
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm_1 = nn.LSTMCell(input_size=embedding_dim, hidden_size=hidden_dim)
        self.lstm_2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim) 
        
        self.dropout = nn.Dropout(p=0.5)

        # fully connected layer to connect the output of the LSTM cell to the output
        self.fc = nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        
    def forward(self, x, hc, return_hc=False):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        
        # empty tensor for the output of the lstm
        output_seq = torch.empty((seq_len, batch_size, self.vocab_size))
        output_seq = gpu(output_seq)
        hc1, hc2 = hc, hc

        # for every step in the sequence
        for t in range(seq_len):
            out_t, hc1, hc2 = self.feed_one_x_t(x[t], hc1, hc2)
            output_seq[t] = out_t
        
        if return_hc:
            return output_seq, hc1, hc2
        return output_seq
            
    def init_hidden(self, bs=None):
        if bs is None:
            bs = self.batch_size
        # initialize the <hidden state> and the <cell state> to zeros
        return (gpu(torch.zeros(bs, self.hidden_dim)), gpu(torch.zeros(bs, self.hidden_dim)))
    
    def feed_one_x_t(self, x_t, hc1, hc2):
        # convert batch of single ints to batch of embeddings
        xt_emb = self.emb(x_t) # returns (batch_size, embedding_dim)

        # get the hidden and cell states from the first layer cell
        hc1 = self.lstm_1(xt_emb, hc1)
        h1, c1 = hc1 # unpack the hidden and the cell states from the first layer

        # pass the hidden state from the first layer to the cell in the second layer
        hc2 = self.lstm_2(h1, hc2)
        h2, c2 = hc2 # unpack the hidden and cell states from the second layer cell

        # form the output of the fc
        out_t = self.fc(self.dropout(h2))
        
        return out_t, hc1, hc2
    
    def feed_one_char(self, char, hc1, hc2):
        ints = [self.char2int[char]] # sequence of ints 
        ints = [ints] # a 1-batch of seqs
        x = ints_to_tensor(ints) # shape of (seq_len, batch_size)
        x_t = x[0] # take the first (single) part of the sequence
        
        return self.feed_one_x_t(x_t, hc1, hc2)
    
    def warm_up(self, base_str):
        hc = self.init_hidden(bs=1)
        ints = [self.char2int[c] for c in base_str]  # sequence of ints 
        ints = [ints] # a 1-batch of seqs
        x = ints_to_tensor(ints) # shape of (seq_len, batch_size)
        
        out, hc1, hc2 = self.forward(x, hc, return_hc=True)
        return out, hc1, hc2
    
    def sample_char(self, out_t, top_k=5):
        # apply the softmax to the output to get the probabilities of the characters
        out_t = F.softmax(out_t, dim=1)

        # out_t now holds the vector of predictions (1, vocab_size)
        # we want to sample 5 top characters
        p, top_char = out_t.topk(top_k) # returns tuple (top_values, top_indices)

        # get the top k characters by their probabilities
        top_char = top_char.cpu().squeeze().numpy()

        # sample a character using its probability
        p = p.detach().cpu().squeeze().numpy()
        char_int = np.random.choice(top_char, p = p/p.sum())
        
        return self.int2char[char_int]
        
    def predict(self, base_str, top_k=5, seq_len=128):
        self.eval()

        res = np.empty(seq_len+len(base_str), dtype="object")
        for i, c in enumerate(base_str):
            res[i] = c
        
        out_warm, hc1, hc2 = self.warm_up(base_str)
        out_t = out_warm[-1]

        for i in range(seq_len):
            char = self.sample_char(out_t, top_k)
            out_t, hc1, hc2 = self.feed_one_char(char, hc1, hc2)
            res[i + len(base_str)] = char
        
        return ''.join(res)
        

def load():
    state = torch.load("save_2", map_location="cpu")
    net = state["net"]
    char2int = state["char2int"]
    int2char = state["int2char"]
    net.char2int = char2int
    net.int2char = int2char
    return state

def save(state):
    torch.save(state, "save_2")
