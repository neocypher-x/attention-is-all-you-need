import unittest
import torch

from transformer import *

class TestTokenEmbedding(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.vocab_size = 100
        self.d_model = 16
        self.batch_size = 4
        self.seq_len = 10
        self.embedding_layer = TokenEmbedding(self.vocab_size, self.d_model)

    def test_output_shape(self):
        """Test if TokenEmbedding outputs the correct shape."""
        test_input = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = self.embedding_layer(test_input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Unexpected shape: {output.shape}")

    def test_output_type(self):
        """Test if output is a torch tensor of type float32."""
        test_input = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = self.embedding_layer(test_input)
        self.assertIsInstance(output, torch.Tensor, "Output is not a tensor")
        self.assertEqual(output.dtype, torch.float32, f"Unexpected dtype: {output.dtype}")

    def test_same_index_same_embedding(self):
        """Test if the same token index maps to the same embedding."""
        index = torch.tensor([[5]])
        embedding_1 = self.embedding_layer(index)
        embedding_2 = self.embedding_layer(index)
        self.assertTrue(torch.allclose(embedding_1, embedding_2),
                        "Embeddings should be identical for the same index")

    def test_different_indices_different_embeddings(self):
        """Test if different token indices map to different embeddings."""
        index1 = torch.tensor([[5]])
        index2 = torch.tensor([[8]])
        embedding_1 = self.embedding_layer(index1)
        embedding_2 = self.embedding_layer(index2)
        self.assertFalse(torch.allclose(embedding_1, embedding_2),
                         "Different indices should have different embeddings")

    def test_gradient_computation(self):
        """Test if gradients are computed correctly."""
        test_input = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        output = self.embedding_layer(test_input)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(self.embedding_layer.embedding.weight.grad, "Gradients should not be None")
        self.assertEqual(self.embedding_layer.embedding.weight.grad.shape, (self.vocab_size, self.d_model),
                         "Gradient shape mismatch")

class TestPositionalEncoding(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.d_model = 16
        self.seq_len = 10
        self.batch_size = 4
        self.test_input = torch.zeros((self.batch_size, self.seq_len, self.d_model))  # Placeholder embeddings
        self.pos_encoding = PositionalEncoding(d_model=self.d_model)

    def test_output_shape(self):
        """Test if PositionalEncoding outputs the correct shape."""
        output = self.pos_encoding(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Unexpected shape: {output.shape}")

    def test_output_type(self):
        """Test if output is a torch tensor of type float32."""
        output = self.pos_encoding(self.test_input)
        self.assertIsInstance(output, torch.Tensor, "Output is not a tensor")
        self.assertEqual(output.dtype, torch.float32, f"Unexpected dtype: {output.dtype}")

    def test_positional_encoding_added(self):
        """Test if Positional Encoding modifies the input embeddings."""
        output = self.pos_encoding(self.test_input)
        self.assertFalse(torch.allclose(self.test_input, output),
                         "Positional encoding is not being added!")

    def test_device_compatibility(self):
        """Test if PositionalEncoding works on both CPU and GPU."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_input = self.test_input.to(device)
        pos_encoding = self.pos_encoding.to(device)
        output = pos_encoding(test_input)
        self.assertEqual(output.device, test_input.device,
                         f"Device mismatch: {output.device} vs {test_input.device}")

    def test_deterministic_encoding(self):
        """Test if PositionalEncoding is deterministic (same input → same output)."""
        output1 = self.pos_encoding(self.test_input)
        output2 = self.pos_encoding(self.test_input)
        self.assertTrue(torch.allclose(output1, output2),
                        "Positional encoding should be deterministic!")

class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        torch.manual_seed(42)  # Ensure reproducibility

        self.batch_size = 2
        self.num_heads = 3
        self.seq_len_q = 4  # Query sequence length
        self.seq_len_k = 5  # Key sequence length
        self.d_k = 6        # Dimension per head for query/key
        self.d_v = 6        # Dimension per head for value

        # Create a ScaledDotProductAttention instance
        self.attention_module = ScaledDotProductAttention(self.d_k)

        # Create random query, key, value
        self.query = torch.randn(self.batch_size, self.num_heads, self.seq_len_q, self.d_k)
        self.key = torch.randn(self.batch_size, self.num_heads, self.seq_len_k, self.d_k)
        self.value = torch.randn(self.batch_size, self.num_heads, self.seq_len_k, self.d_v)

    def test_forward_pass_without_mask(self):
        """Test forward pass without a mask."""
        output, attn_weights = self.attention_module(self.query, self.key, self.value, mask=None)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.seq_len_q, self.d_v),
                         f"Output shape mismatch. Got {output.shape}")

        # Check attention weight shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.seq_len_q, self.seq_len_k),
                         f"Attention weights shape mismatch. Got {attn_weights.shape}")

        # Check attention weights sum to ~1 across the last dimension
        attn_sum = attn_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5),
                        "Attention weights do not sum to 1 along the last dimension.")

    def test_forward_pass_with_mask(self):
        """Test forward pass with a mask."""
        mask = torch.ones(self.batch_size, 1, self.seq_len_q, self.seq_len_k)
        mask[:, :, :, -2:] = 0  # Mask out the last 2 positions

        output_masked, attn_weights_masked = self.attention_module(self.query, self.key, self.value, mask=mask)

        # Check that masked positions in softmax are effectively 0
        masked_positions = attn_weights_masked[..., -2:]  # Last two positions
        self.assertTrue(torch.allclose(masked_positions, torch.zeros_like(masked_positions), atol=1e-5),
                        "Masking does not appear to zero out the last two positions.")

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = 16
        self.num_heads = 4

        # Initialize test input tensors
        self.query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.value = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Initialize multi-head attention module
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)

    def test_output_shape(self):
        """Test if MultiHeadAttention outputs the correct shape."""
        output = self.mha(self.query, self.key, self.value, mask=None)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Unexpected output shape: {output.shape}")

    def test_output_type(self):
        """Test if output is a torch tensor."""
        output = self.mha(self.query, self.key, self.value, mask=None)
        self.assertIsInstance(output, torch.Tensor, "Output is not a tensor")

    def test_deterministic_output(self):
        """Test if MultiHeadAttention is deterministic (same input → same output)."""
        output_1 = self.mha(self.query, self.key, self.value, mask=None)
        output_2 = self.mha(self.query, self.key, self.value, mask=None)
        self.assertTrue(torch.allclose(output_1, output_2),
                        "Output should be deterministic!")

    def test_masking(self):
        """Test if MultiHeadAttention handles masking correctly."""
        mask = torch.zeros(self.batch_size, 1, self.seq_len, self.seq_len)
        mask[:, :, :, -1] = float('-inf')  # Mask the last token

        output_masked = self.mha(self.query, self.key, self.value, mask=mask)

        # Ensure output is still the correct shape
        self.assertEqual(output_masked.shape, (self.batch_size, self.seq_len, self.d_model),
                         "Masked output shape mismatch")

class TestPositionwiseFeedForward(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.batch_size = 2
        self.seq_len = 5
        self.d_model = 16
        self.d_ff = 32  # Expanded hidden dimension

        # Initialize test input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Initialize PositionwiseFeedForward module
        self.ffn = PositionwiseFeedForward(self.d_model, self.d_ff)

    def test_output_shape(self):
        """Test if PositionwiseFeedForward outputs the correct shape."""
        output = self.ffn(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Unexpected output shape: {output.shape}")

    def test_output_type(self):
        """Test if output is a torch tensor."""
        output = self.ffn(self.x)
        self.assertIsInstance(output, torch.Tensor, "Output is not a tensor")

    def test_relu_activation(self):
        """Test if ReLU activation is correctly applied."""
        hidden_layer_output = self.ffn.fc1(self.x)  # Get pre-ReLU values
        self.assertTrue(torch.all((hidden_layer_output > 0) == (self.ffn.relu(hidden_layer_output) > 0)),
                        "ReLU activation is not applied correctly")

    def test_deterministic_output(self):
        """Test if PositionwiseFeedForward is deterministic (same input → same output)."""
        output_1 = self.ffn(self.x)
        output_2 = self.ffn(self.x)
        self.assertTrue(torch.allclose(output_1, output_2),
                        "Output should be deterministic!")

    def test_gradient_computation(self):
        """Test if gradients are computed correctly."""
        output = self.ffn(self.x)
        output.sum().backward()  # Compute gradients
        self.assertIsNotNone(self.ffn.fc1.weight.grad, "Gradients are not computed for fc1!")
        self.assertIsNotNone(self.ffn.fc2.weight.grad, "Gradients are not computed for fc2!")

class TestTransformerEncoderLayer(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.d_model = 32
        self.num_heads = 4
        self.d_ff = 64
        self.dropout = 0.1
        self.batch_size = 2
        self.seq_len = 5

        # Initialize TransformerEncoderLayer
        self.encoder_layer = TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)

        # Dummy input
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Mask: a binary mask allowing everything (all 1s).
        self.mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)

    def test_instantiation(self):
        """Test if TransformerEncoderLayer can be instantiated."""
        try:
            TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        except Exception as e:
            self.fail(f"Instantiation failed with error: {e}")

    def test_forward_pass_no_mask(self):
        """Test forward pass without a mask and check output shape."""
        try:
            output_no_mask = self.encoder_layer(self.x)  # No mask
        except Exception as e:
            self.fail(f"Forward pass failed without mask: {e}")

        self.assertEqual(output_no_mask.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape {output_no_mask.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_forward_pass_with_mask(self):
        """Test forward pass with a mask and check output shape."""
        try:
            output_with_mask = self.encoder_layer(self.x, mask=self.mask)
        except Exception as e:
            self.fail(f"Forward pass failed with mask: {e}")

        self.assertEqual(output_with_mask.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape {output_with_mask.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_relu_activation(self):
        """Check if the TransformerEncoderLayer contains ReLU activation in FFN."""
        found_relu = any(isinstance(submodule, nn.ReLU) for submodule in self.encoder_layer.modules())
        self.assertTrue(found_relu,
                        "No ReLU found in TransformerEncoderLayer's FFN. If using a different activation, adjust this test.")

    def test_gradient_backprop(self):
        """Test if gradients propagate correctly during backpropagation."""
        output = self.encoder_layer(self.x, mask=self.mask)
        output.sum().backward()  # Compute gradients

        # Ensure gradients exist for at least one parameter
        has_gradients = any(param.grad is not None for param in self.encoder_layer.parameters())
        self.assertTrue(has_gradients, "No gradients computed during backpropagation!")

class TestTransformerDecoderLayer(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.d_model = 32
        self.num_heads = 4
        self.d_ff = 64
        self.dropout = 0.1
        self.batch_size = 2
        self.seq_len = 5

        # Instantiate TransformerDecoderLayer
        self.decoder_layer = TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)

        # Dummy input tensors
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)  # Decoder input
        self.memory = torch.randn(self.batch_size, self.seq_len, self.d_model)  # Encoder output

        # Dummy masks
        self.src_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)  # Source mask
        self.tgt_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)  # Target mask

    def test_instantiation(self):
        """Test if TransformerDecoderLayer can be instantiated."""
        try:
            TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        except Exception as e:
            self.fail(f"Instantiation failed with error: {e}")

    def test_forward_pass_no_masks(self):
        """Test forward pass without masks and check output shape."""
        try:
            output = self.decoder_layer(self.x, self.memory)
        except Exception as e:
            self.fail(f"Forward pass failed without masks: {e}")

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape {output.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_forward_pass_with_masks(self):
        """Test forward pass with source and target masks."""
        try:
            output = self.decoder_layer(self.x, self.memory, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        except Exception as e:
            self.fail(f"Forward pass failed with masks: {e}")

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape with masks {output.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_relu_activation(self):
        """Check if the TransformerDecoderLayer contains ReLU activation in FFN."""
        found_relu = any(isinstance(module, nn.ReLU) for module in self.decoder_layer.modules())
        self.assertTrue(found_relu, "No ReLU found in TransformerDecoderLayer (if you expected one).")

    def test_gradient_backprop(self):
        """Test if gradients propagate correctly during backpropagation."""
        output = self.decoder_layer(self.x, self.memory, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        output.sum().backward()  # Compute gradients

        # Ensure gradients exist for at least one parameter
        has_gradients = any(param.grad is not None for param in self.decoder_layer.parameters())
        self.assertTrue(has_gradients, "No gradients computed during backpropagation!")

class TestTransformerEncoder(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.num_layers = 2
        self.d_model = 32
        self.num_heads = 4
        self.d_ff = 64
        self.dropout = 0.1
        self.batch_size = 2
        self.seq_len = 5

        # Instantiate TransformerEncoder
        self.encoder = TransformerEncoder(self.num_layers, self.d_model, self.num_heads, self.d_ff, self.dropout)

        # Dummy input tensor
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)

        # Dummy mask (all ones, placeholder)
        self.mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)

    def test_instantiation(self):
        """Test if TransformerEncoder can be instantiated."""
        try:
            TransformerEncoder(self.num_layers, self.d_model, self.num_heads, self.d_ff, self.dropout)
        except Exception as e:
            self.fail(f"Instantiation failed with error: {e}")

    def test_forward_pass_no_mask(self):
        """Test forward pass without a mask and check output shape."""
        try:
            output_no_mask = self.encoder(self.x)  # No mask
        except Exception as e:
            self.fail(f"Forward pass failed without mask: {e}")

        self.assertEqual(output_no_mask.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape {output_no_mask.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_forward_pass_with_mask(self):
        """Test forward pass with a mask and check output shape."""
        try:
            output_with_mask = self.encoder(self.x, self.mask)
        except Exception as e:
            self.fail(f"Forward pass failed with mask: {e}")

        self.assertEqual(output_with_mask.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape with mask {output_with_mask.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_relu_activation(self):
        """Check if TransformerEncoder contains ReLU activation in FFN."""
        found_relu = any(isinstance(module, nn.ReLU) for module in self.encoder.modules())
        self.assertTrue(found_relu, "No ReLU found in TransformerEncoder (if you expected one).")

    def test_gradient_backprop(self):
        """Test if gradients propagate correctly during backpropagation."""
        output = self.encoder(self.x, self.mask)
        output.sum().backward()  # Compute gradients

        # Ensure gradients exist for at least one parameter
        has_gradients = any(param.grad is not None for param in self.encoder.parameters())
        self.assertTrue(has_gradients, "No gradients computed during backpropagation!")

class TestTransformerDecoder(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.num_layers = 2
        self.d_model = 32
        self.num_heads = 4
        self.d_ff = 64
        self.dropout = 0.1
        self.batch_size = 2
        self.seq_len = 5

        # Instantiate TransformerDecoder
        self.decoder = TransformerDecoder(self.num_layers, self.d_model, self.num_heads, self.d_ff, self.dropout)

        # Dummy input tensors
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model)      # Target sequence
        self.memory = torch.randn(self.batch_size, self.seq_len, self.d_model) # Encoder output

        # Dummy masks
        self.src_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)  # Encoder mask
        self.tgt_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)  # Decoder mask

    def test_instantiation(self):
        """Test if TransformerDecoder can be instantiated."""
        try:
            TransformerDecoder(self.num_layers, self.d_model, self.num_heads, self.d_ff, self.dropout)
        except Exception as e:
            self.fail(f"Instantiation failed with error: {e}")

    def test_forward_pass_no_masks(self):
        """Test forward pass without masks and check output shape."""
        try:
            output_no_mask = self.decoder(self.x, self.memory)
        except Exception as e:
            self.fail(f"Forward pass failed without masks: {e}")

        self.assertEqual(output_no_mask.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape {output_no_mask.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_forward_pass_with_masks(self):
        """Test forward pass with source and target masks."""
        try:
            output_with_masks = self.decoder(self.x, self.memory, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        except Exception as e:
            self.fail(f"Forward pass failed with masks: {e}")

        self.assertEqual(output_with_masks.shape, (self.batch_size, self.seq_len, self.d_model),
                         f"Output shape with masks {output_with_masks.shape} != {(self.batch_size, self.seq_len, self.d_model)}")

    def test_relu_activation(self):
        """Check if TransformerDecoder contains ReLU activation in FFN."""
        found_relu = any(isinstance(module, nn.ReLU) for module in self.decoder.modules())
        self.assertTrue(found_relu, "No ReLU found in TransformerDecoder (if you expected one).")

    def test_gradient_backprop(self):
        """Test if gradients propagate correctly during backpropagation."""
        output = self.decoder(self.x, self.memory, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        output.sum().backward()  # Compute gradients

        # Ensure gradients exist for at least one parameter
        has_gradients = any(param.grad is not None for param in self.decoder.parameters())
        self.assertTrue(has_gradients, "No gradients computed during backpropagation!")

class TestTransformer(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.vocab_size = 10
        self.d_model = 8
        self.num_layers = 2
        self.num_heads = 2
        self.d_ff = 16
        self.dropout = 0.1
        self.batch_size = 2
        self.src_seq_len = 5
        self.tgt_seq_len = 6

        # Instantiate Transformer model
        self.model = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout
        )

        # Dummy input token indices
        self.src = torch.randint(0, self.vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt = torch.randint(0, self.vocab_size, (self.batch_size, self.tgt_seq_len))

        # Dummy masks
        self.src_mask = None  # No source mask
        self.tgt_mask = torch.ones(self.batch_size, 1, self.tgt_seq_len, self.tgt_seq_len)  # Target self-attention mask

    def test_instantiation(self):
        """Test if Transformer can be instantiated."""
        try:
            Transformer(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )
        except Exception as e:
            self.fail(f"Instantiation failed with error: {e}")

    def test_forward_pass_no_masks(self):
        """Test forward pass without masks and check output shape."""
        try:
            output_no_mask = self.model(self.src, self.tgt)  # No masks
        except Exception as e:
            self.fail(f"Forward pass failed without masks: {e}")

        expected_shape = (self.batch_size, self.tgt_seq_len, self.vocab_size)
        self.assertEqual(output_no_mask.shape, expected_shape,
                         f"Output shape {output_no_mask.shape} != {expected_shape}")

    def test_forward_pass_with_tgt_mask(self):
        """Test forward pass with a target mask and check output shape."""
        try:
            output_with_masks = self.model(self.src, self.tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        except Exception as e:
            self.fail(f"Forward pass failed with masks: {e}")

        expected_shape = (self.batch_size, self.tgt_seq_len, self.vocab_size)
        self.assertEqual(output_with_masks.shape, expected_shape,
                         f"Output shape with masks {output_with_masks.shape} != {expected_shape}")

    def test_relu_activation(self):
        """Check if Transformer contains ReLU activation in FFN."""
        found_relu = any(isinstance(module, nn.ReLU) for module in self.model.modules())
        self.assertTrue(found_relu, "No ReLU found in Transformer (if you expected it).")

    def test_gradient_backprop(self):
        """Test if gradients propagate correctly during backpropagation."""
        output = self.model(self.src, self.tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask)
        loss = output.sum()  # Simple scalar loss
        try:
            loss.backward()
        except Exception as e:
            self.fail(f"Backward pass failed: {e}")

class TestTransformerTrainer(unittest.TestCase):
    def setUp(self):
        """Initialize common test variables."""
        self.vocab_size = 10
        self.d_model = 8
        self.num_layers = 2
        self.num_heads = 2
        self.d_ff = 16
        self.dropout = 0.1
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 2
        self.src_seq_len = 5
        self.tgt_seq_len = 6

        # Instantiate Transformer model
        self.model = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout=self.dropout
        )

        # Instantiate TransformerTrainer
        self.trainer = TransformerTrainer(self.model, self.learning_rate, self.weight_decay)

        # Dummy input tensors
        self.src = torch.randint(0, self.vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt = torch.randint(0, self.vocab_size, (self.batch_size, self.tgt_seq_len))

    def test_instantiation(self):
        """Test if TransformerTrainer can be instantiated."""
        try:
            TransformerTrainer(self.model, self.learning_rate, self.weight_decay)
        except Exception as e:
            self.fail(f"Instantiation of TransformerTrainer failed with error: {e}")

    def test_train_step(self):
        """Test if train_step runs without errors and returns a scalar loss."""
        try:
            loss = self.trainer.train_step(self.src, self.tgt)
        except Exception as e:
            self.fail(f"train_step failed with error: {e}")

        self.assertIsInstance(loss, torch.Tensor, f"train_step should return a torch.Tensor, got {type(loss)}")
        self.assertEqual(loss.dim(), 0, f"Loss should be a 0-dimensional scalar tensor, got shape {loss.shape}")

    def test_evaluate(self):
        """Test if evaluate runs without errors and returns a numeric score."""
        try:
            score = self.trainer.evaluate(self.src, self.tgt)
        except Exception as e:
            self.fail(f"evaluate failed with error: {e}")

        self.assertIsInstance(score, float, f"evaluate should return a float, got {type(score)}")


if __name__ == "__main__":
    unittest.main()
