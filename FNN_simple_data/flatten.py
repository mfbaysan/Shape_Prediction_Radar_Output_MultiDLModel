import torch
import math

def positionalencoding1d(d_model, sample):
  """
  :param d_model: dimension of the model
  :param length: length of positions
  :return: length*d_model position matrix
  """
  if d_model % 2 != 0:
    raise ValueError("Cannot use sin/cos positional encoding with "
                     "odd dim (got dim={:d})".format(d_model))
  pe = torch.zeros(sample.shape[0], d_model)
  position = sample.unsqueeze(1)
  div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model)))
  pe[:, 0::2] = torch.sin(position.float() * div_term)
  pe[:, 1::2] = torch.cos(position.float() * div_term)

  pe = pe.permute(1, 0)
  return pe


def apply_positional_encoding(x, length):

  batch_size, seq_len, d_model = x.shape  # Extract dimensions
  pe = positionalencoding1d(d_model, length)  # Generate positional encoding

  # Expand positional encoding to match sequence dimension
  pe = pe.unsqueeze(0).expand(batch_size, -1, seq_len)  # Add batch dimension and expand

  # Add positional encoding to sequences (element-wise addition)
  encoded_x = x + pe

  return encoded_x

# Example usage
x = torch.randn(512)  # Assuming batch size 32, seq len 1, embedding dim 512
length = 512  # Desired sequence length for positional encoding

encoded_x = positionalencoding1d(6, x)
print(encoded_x.shape)  # Output: torch.Size([32, 512, 10])

