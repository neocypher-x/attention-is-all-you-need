import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes the embedding layer.

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            d_model (int): Dimension of the embedding vectors.
        """
        super().__init__()
        
        # TODO: Define the embedding layer that maps token indices to dense vectors.
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for token embedding.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len) containing token indices.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len, d_model) containing embedded representations.
        """
        # TODO: Implement the lookup operation using the embedding layer.
        embedded = self.embedding(x)  

        return embedded
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Create a (max_len, d_model) tensor to hold the positional encodings
        pe = torch.zeros(max_len, d_model)            # shape: (max_len, d_model)
        
        # position: shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: shape (d_model/2,)  -> we’ll use it for the even/odd splits
        # This follows exp(- log(10000) * (2i/d_model)) = 10000^(-2i/d_model).
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer so it's not trained
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)

        # Grab up to seq_len positions from pe and add to x
        # shape of pe_slice becomes (1, seq_len, d_model)
        pe_slice = self.pe[:seq_len, :].unsqueeze(0).to(x.device)

        return x + pe_slice
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super().__init__()
        self.d_k = d_k   # for scaling

    def forward(
        self,
        query: torch.Tensor,   # (batch_size, num_heads, seq_len, d_k)
        key: torch.Tensor,     # (batch_size, num_heads, seq_len, d_k)
        value: torch.Tensor,   # (batch_size, num_heads, seq_len, d_v)
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (1) QK^T
        # key.transpose(-2, -1) is shape (batch_size, num_heads, d_k, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # (2) Scale by sqrt(d_k)
        attention_scores = attention_scores / math.sqrt(self.d_k)

        # (3) If mask is provided, set masked positions to -inf
        if mask is not None:
            # Typically a 1/0 mask is used; we want to fill 0’s with -inf
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # (4) Apply softmax over the last dimension (seq_len of the key)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # (5) Multiply by V
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Initializes multi-head attention.

        Args:
            d_model (int): Dimension of the model (input and output size).
            num_heads (int): Number of attention heads.
        """
        super().__init__()

        # TODO: Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # TODO: Define linear transformations for query, key, and value
        self.W_q = nn.Linear(d_model, d_model)  # Replace with nn.Linear
        self.W_k = nn.Linear(d_model, d_model)  # Replace with nn.Linear
        self.W_v = nn.Linear(d_model, d_model)  # Replace with nn.Linear

        # TODO: Define output projection layer
        self.W_o = nn.Linear(d_model, d_model)  # Replace with nn.Linear

        # TODO: Define the scaled dot-product attention module
        self.attention = ScaledDotProductAttention(self.d_k)  # Replace with ScaledDotProductAttention(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes multi-head attention.

        Args:
            query (torch.Tensor): Shape (batch_size, seq_len, d_model)
            key (torch.Tensor): Shape (batch_size, seq_len, d_model)
            value (torch.Tensor): Shape (batch_size, seq_len, d_model)
            mask (Optional[torch.Tensor]): Shape (batch_size, 1, seq_len, seq_len)

        Returns:
            torch.Tensor: Shape (batch_size, seq_len, d_model) - Multi-head attention output.
        """
        # TODO: Apply linear transformations to query, key, and value
        Q = self.W_q(query)  # Replace with correct transformation
        K = self.W_k(key)  # Replace with correct transformation
        V = self.W_v(value)  # Replace with correct transformation

        # after applying the linear layers
        # Q.shape = (B, L_q, d_model)
        # K.shape = (B, L_k, d_model)
        # V.shape = (B, L_v, d_model)

        B, L_q, _ = Q.shape
        B_k, L_k, _ = K.shape
        B_v, L_v, _ = V.shape

        # TODO: Reshape Q, K, V for multi-head attention
        # Hint: Use `.view()` and `.transpose()` to shape into (batch_size, num_heads, seq_len, d_k)
        batch_size, seq_len, _ = query.shape
        # They should match in batch dimension:
        assert B == B_k == B_v, "Batch sizes must match!"

        Q = Q.view(B, L_q, self.num_heads, self.d_k).transpose(1,2)  # => (B, num_heads, L_q, d_k)
        K = K.view(B, L_k, self.num_heads, self.d_k).transpose(1,2)  # => (B, num_heads, L_k, d_k)
        V = V.view(B, L_v, self.num_heads, self.d_k).transpose(1,2)  # => (B, num_heads, L_v, d_k)

        # Then feed to ScaledDotProductAttention
        # TODO: Apply scaled dot-product attention
        output, attention_weights = self.attention(Q, K, V, mask)  # Replace with correct computation

        # TODO: Concatenate the heads back and apply final linear transformation
        # Current shape: (batch_size, num_heads, seq_len, d_k)
        # We first swap num_heads and seq_len
        output = output.transpose(1, 2)  # (batch_size, seq_len, num_heads, d_k)
        output = output.contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(output)  # Replace with correct transformation

        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            (batch_size, seq_len, d_model) - Transformed representations.
        """
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)

        return output
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a single Transformer Encoder Layer.

        Args:
            d_model (int): The embedding dimension (must be divisible by num_heads).
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size of the feed-forward network.
            dropout (float): Dropout rate (default 0.1).
        """
        super().__init__()

        # TODO: Define multi-head self-attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # Replace with MultiHeadAttention(d_model, num_heads)

        # TODO: Define feed-forward network (FFN)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)  # Replace with a two-layer FFN

        # TODO: Define Layer Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 

        # TODO: Define Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Transformer Encoder Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Mask for attention (default None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # TODO: Apply multi-head self-attention
        attn_output = self.self_attn(x, x, x, mask=mask)

        # TODO: Apply residual connection and layer normalization
        x = self.norm1(x + self.dropout1(attn_output))

        # TODO: Apply feed-forward network
        ffn_output = self.ffn(x) 

        # TODO: Apply second residual connection and layer normalization
        x = self.norm2(x + self.dropout2(ffn_output))

        return x
    
class TransformerDecoderLayer(nn.Module): 
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        """
        Initializes a single Transformer Decoder Layer.

        Args:
            d_model (int): The embedding dimension (must be divisible by num_heads).
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size of the feed-forward network.
            dropout (float): Dropout rate (default 0.1).
        """
        super().__init__()

        # TODO: Define masked multi-head self-attention layer
        self.self_attn = MultiHeadAttention(d_model, num_heads)

        # TODO: Define multi-head attention layer for encoder-decoder attention
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)  # Replace with MultiHeadAttention(d_model, num_heads)

        # TODO: Define feed-forward network (FFN)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        # TODO: Define Layer Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.norm3 = nn.LayerNorm(d_model) 

        # TODO: Define Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Transformer Decoder Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model) (decoder input).
            memory (torch.Tensor): Encoder outputs of shape (batch_size, seq_len_enc, d_model).
            tgt_mask (Optional[torch.Tensor]): Mask for target self-attention (default None).
            src_mask (Optional[torch.Tensor]): Mask for encoder-decoder attention (default None).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # TODO: Apply masked multi-head self-attention
        attn_output = self.self_attn(x, x, x, mask=tgt_mask) 

        # TODO: Apply residual connection and layer normalization
        x = self.norm1(x + self.dropout1(attn_output))

        # TODO: Apply encoder-decoder multi-head attention
        attn_output_2 = self.enc_dec_attn(query=x, key=memory, value=memory, mask=src_mask)

        # TODO: Apply residual connection and layer normalization
        x = self.norm2(x + self.dropout2(attn_output_2))

        # TODO: Apply feed-forward network
        ffn_output = self.ffn(x)

        # TODO: Apply final residual connection and layer normalization
        x = self.norm3(x + self.dropout3(ffn_output))

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a Transformer Encoder consisting of multiple encoder layers.

        Args:
            num_layers (int): Number of TransformerEncoderLayer layers.
            d_model (int): Dimension of embeddings and model size.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size in feed-forward network.
            dropout (float): Dropout rate (default 0.1).
        """
        super().__init__()

        # TODO: Define a stack of TransformerEncoderLayers
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  

        # TODO: Define final layer normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer Encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[torch.Tensor]): Optional mask for attention.

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, seq_len, d_model).
        """
        # TODO: Pass input through each TransformerEncoderLayer
        for layer in self.layers:
            x = layer(x, mask)

        # TODO: Apply final normalization
        x = self.norm(x)  

        return x
import torch
import torch.nn as nn
from typing import Optional

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a Transformer Decoder consisting of multiple decoder layers.

        Args:
            num_layers (int): Number of TransformerDecoderLayer layers.
            d_model (int): Dimension of embeddings and model size.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size in feed-forward network.
            dropout (float): Dropout rate (default 0.1).
        """
        super().__init__()

        # TODO: Define a stack of TransformerDecoderLayers
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])  # Replace with nn.ModuleList([...])

        # TODO: Define final layer normalization
        self.norm = nn.LayerNorm(d_model)  # Replace with nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer Decoder.

        Args:
            x (torch.Tensor): Decoder input tensor (batch_size, tgt_seq_len, d_model).
            memory (torch.Tensor): Encoder outputs (batch_size, src_seq_len, d_model).
            src_mask (Optional[torch.Tensor]): Mask for encoder-decoder attention.
            tgt_mask (Optional[torch.Tensor]): Mask for target self-attention.

        Returns:
            torch.Tensor: Decoded representation of shape (batch_size, tgt_seq_len, d_model).
        """
        # TODO: Pass input through each TransformerDecoderLayer
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)  # Replace with layer(x, memory, src_mask, tgt_mask)

        # TODO: Apply final normalization
        x = self.norm(x)  # Replace with self.norm(x)

        return x
class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, dropout: float):
        """
        Initializes the Transformer model.

        Args:
            vocab_size (int): Number of unique tokens in the vocabulary.
            d_model (int): Embedding dimension.
            num_layers (int): Number of encoder and decoder layers.
            num_heads (int): Number of attention heads.
            d_ff (int): Hidden layer size in feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.d_model = d_model

        # TODO: Define token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)  # Replace with nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # TODO: Define positional encoding
        self.pos_encoding = PositionalEncoding(d_model)  # Replace with PositionalEncoding(d_model)

        # TODO: Define the encoder
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)  # Replace with TransformerEncoder(...)

        # TODO: Define the decoder
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)  # Replace with TransformerDecoder(...)

        # TODO: Define the final projection layer
        self.fc_out = nn.Linear(d_model, vocab_size)  # Replace with nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            src (torch.Tensor): Source token indices of shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target token indices of shape (batch_size, tgt_seq_len).
            src_mask (Optional[torch.Tensor]): Source mask of shape (batch_size, 1, src_seq_len, src_seq_len).
            tgt_mask (Optional[torch.Tensor]): Target mask of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).

        Returns:
            torch.Tensor: Token probabilities of shape (batch_size, tgt_seq_len, vocab_size).
        """
        # TODO: Apply token embedding and positional encoding
        """
        The original Attention Is All You Need paper did this scaling to keep the variance of the embeddings roughly on par with the variance of the positional encodings.
        Doing it before positional encoding ensures that the learned embeddings and the fixed sin/cos signals are on a compatible magnitude scale.
        Typically, you also apply dropout to the sum of Embeddings+PositionalEncoding.
        """
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)     # optional dropout here if you're following the paper closely

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # TODO: Pass through the encoder
        memory = self.encoder(src_emb, src_mask)  # Replace with self.encoder(src_emb, src_mask)

        # TODO: Pass through the decoder
        output = self.decoder(tgt_emb, memory, src_mask, tgt_mask)  # Replace with self.decoder(tgt_emb, memory, src_mask, tgt_mask)

        # TODO: Apply final linear layer to project into vocab size
        logits = self.fc_out(output)  # Replace with self.fc_out(output)

        return logits

class TransformerTrainer:
    def __init__(self, model: nn.Module, learning_rate: float, weight_decay: float):
        """
        Initializes optimizer and loss function for training.

        Args:
            model (nn.Module): A Transformer model (e.g., your 'Transformer' class).
            learning_rate (float): Learning rate for optimizer.
            weight_decay (float): Weight decay (L2 regularization) factor.
        """
        self.model = model
        
        # Typically we use Adam or AdamW for Transformers
        self.optimizer = optim.AdamW(model.parameters(), 
                                     lr=learning_rate, 
                                     weight_decay=weight_decay)
        
        # Commonly cross-entropy (or label smoothing cross-entropy) is used
        # Here we'll just define a basic cross-entropy as an example
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Runs a single training step (forward + backward + optimizer step).

        Args:
            src (torch.Tensor): Source sequence indices, shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target sequence indices, shape (batch_size, tgt_seq_len).

        Returns:
            torch.Tensor: Scalar loss (0-dimensional tensor).
        """
        # 1. Switch model to training mode
        self.model.train()

        # 2. Zero out gradients
        self.optimizer.zero_grad()

        # 3. Forward pass
        # Often we shift the `tgt` by one position for teacher forcing; 
        # this is just a minimal stub for demonstration
        logits = self.model(src, tgt)

        # 4. Compute loss
        # Typically we remove the last token or the first token, etc. Here is a simplistic approach:
        # We flatten the predictions and targets to compute cross-entropy
        batch_size, tgt_seq_len, vocab_size = logits.shape
        loss = self.criterion(
            logits.view(batch_size * tgt_seq_len, vocab_size), 
            tgt.view(batch_size * tgt_seq_len)
        )

        # 5. Backprop
        loss.backward()

        # 6. Step optimizer
        self.optimizer.step()

        return loss

    def evaluate(self, src: torch.Tensor, tgt: torch.Tensor) -> float:
        """
        Evaluates the model on a validation set (or any dev set).
        You might compute BLEU, perplexity, etc.

        Args:
            src (torch.Tensor): Source sequence indices, shape (batch_size, src_seq_len).
            tgt (torch.Tensor): Target sequence indices, shape (batch_size, tgt_seq_len).

        Returns:
            float: Example evaluation metric (e.g., BLEU).
        """
        # Switch to eval mode (e.g., disable dropout)
        self.model.eval()

        # For inference or evaluation, you might do something more complex:
        # - generate predictions auto-regressively
        # - compare them to the gold labels (tgt)
        # - compute BLEU or a typical sequence metric

        # We'll just do a simple forward pass + cross-entropy as a placeholder
        with torch.no_grad():
            logits = self.model(src, tgt)
            batch_size, tgt_seq_len, vocab_size = logits.shape
            loss = self.criterion(
                logits.view(batch_size * tgt_seq_len, vocab_size), 
                tgt.view(batch_size * tgt_seq_len)
            )
            
            # Suppose we treat negative loss as "score" for demonstration
            # In reality you'd compute BLEU or some other measure
            dummy_score = float(-loss.item())

        return dummy_score