import torch.nn as nn
import torch
from transformer.attention_head import MultiHeadedAttention, FeedForwardNetwork
from transformer.positional_encodings import PositionalEncodings
import tiktoken

class Transformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.proj = nn.Linear(512, 50_258, device=self.device)
        self.decoder = nn.Sequential(
            nn.Embedding(num_embeddings=50_258, embedding_dim=512, device=self.device),
            PositionalEncodings(512, device=self.device),      
            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512, device=self.device),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512, device=self.device),
                  
            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512, device=self.device),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512, device=self.device),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512, device=self.device),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512, device=self.device),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512, device=self.device),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512, device=self.device),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512, device=self.device),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512, device=self.device),

            MultiHeadedAttention(n_heads=8, d_input=512, d_k=64, device=self.device),     
            nn.LayerNorm(512, device=self.device),
            FeedForwardNetwork(512, device=self.device),
            nn.LayerNorm(512, device=self.device),
        )
        self.tkn = tiktoken.get_encoding("r50k_base")

    def forward(self, X):
        last_token = self.decoder(X)
        output = self.proj(last_token[-1])
        output = torch.softmax(output, dim=0)
        return output
    
    def tokenize_input(self, in_str):
        return self.tkn.encode(in_str)

    def most_likely_token(self, probs):
        return self.tkn.decode([torch.argmax(probs).tolist()])

    def generate_output(self, input_str):

        for i in range(0, 100):
            ctx_encoded = torch.tensor(self.tokenize_input(input_str), device=self.device)
            forwarded = self.forward(ctx_encoded)
            mlt = self.most_likely_token(forwarded)
            input_str += mlt
            print(input_str + "\n\n")
            
        return self.tkn.decode(ctx_encoded.tolist())
    
