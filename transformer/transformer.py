import torch.nn as nn
import torch
from attention_head import MultiHeadedAttention, FeedForwardNetwork

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            Tokenizer(),
            nn.Embedding(),
            PositionalEncoding(),      
            MultiHeadedAttention(),     
            nn.LayerNorm1d(),
            FeedForwardNetwork(),
            nn.LayerNorm1d(),      
        )

        #this takes in raw symnols
        self.decoder = (
            Tokenizer(),
            Embedder(),
            PositionalEncoding(),
            MaskedAttentionHeads(),
        )
        #somewhere here i need to feed in encoder inputs
        self.decoder_encoder = nn.Sequential(
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            AttentionHeads(),
            nn.Linear(),
            nn.Softmax()
        )
