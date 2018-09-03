import torch
from torch import nn
import torch.nn.functional as F

class WordCNN(nn.Module):
    
    def __init__(self, n_classes, vocab, emb, vector_size=300, kernel_sizes=[3,4,5]):
        super(WordCNN, self).__init__()

        vocab_size = vocab.size()
        print('EMB TYPE', type(emb))
        embedding_weight = torch.cuda.FloatTensor(emb)
        self.embedding = nn.Embedding(vocab_size, vector_size)
        self.embedding.weight = nn.Parameter(embedding_weight, requires_grad=False)
        embed_size = vector_size
        
        convs = [nn.Conv1d(in_channels=embed_size, out_channels=100, kernel_size=kernel_size)for kernel_size in kernel_sizes]
        self.conv_modules = nn.ModuleList(convs)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(in_features=300, out_features=n_classes)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, sentences):
        print('WORDCNN SENTENCES:', sentences)
        print('WORDCNN SENTENCES TYPE:', sentences.type())

        # if sentences.type() == torch.cuda.LongTensor:
        if isinstance(sentences, torch.cuda.LongTensor):
            embedded = self.embedding(sentences)
            print('EMBEDDED SIZE:', embedded.size())
        else:
            embedded = torch.mm(sentences.float(), self.embedding.weight)
            print('EMBEDDED SIZE:', embedded.size())
            # embedded = torch.unsqueeze(embedded, 0)
            # print('EMBEDDED SIZE:', embedded.size())

        embedded = embedded.transpose(1,2) # (batch_size, wordvec_size, sentence_length)
        
        feature_list = []
        for conv in self.conv_modules:
            feature_map = self.tanh(conv(embedded))
            max_pooled, argmax = feature_map.max(dim=2)
            feature_list.append(max_pooled)
            
        features = torch.cat(feature_list, dim=1)
        features_regularized = self.dropout(features)
        log_probs = F.log_softmax(self.linear(features_regularized), dim=1)
        # log_probs = self.logsoftmax(self.linear(features_regularized))
        return log_probs