import torch
from torch import nn

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels: int=3,
                 patch_size: int=16,
                 embedding_dim:int=768):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.patcher = nn.Conv2d(in_channels=self.in_channels, out_channels=self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2,  end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        return x_flattened.permute(0, 2, 1)

class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 key_dim: int=768,
                 value_dim=None,
                 attn_dropout:float=0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.key_dim = key_dim
        self.attn_dropout = attn_dropout
        self.value_dim = key_dim if value_dim is None else value_dim

        self.scaling_factor = torch.tensor(self.key_dim)
        self.scaling_factor = torch.sqrt(self.scaling_factor)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.queries = nn.Linear(self.embedding_dim, self.key_dim)   # Queries projection
        self.keys = nn.Linear(self.embedding_dim, self.key_dim)   # Keys projection
        self.values = nn.Linear(self.embedding_dim, self.value_dim)   # Values projection

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x = self.layer_norm(x)

        query = self.queries(x)
        key = self.keys(x)
        value = self.values(x)

        key_transposed = torch.transpose(key, -2, -1)

        attention_scores = torch.matmul(query, key_transposed)
        attention_scores = attention_scores / self.scaling_factor

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value)

        return (attention_output, attention_probs)

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int=768,
                 num_heads: int=12,
                 key_dim: int=768,
                 value_dim=None,
                 attn_dropout:float=0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attn_dropout = attn_dropout
        self.value_dim = key_dim if value_dim is None else value_dim

        self.W_projection = nn.Linear(self.num_heads * self.key_dim, self.embedding_dim)   # Final projection

        self.multi_head_attention = nn.ModuleList([SelfAttentionBlock(self.embedding_dim, self.key_dim, self.value_dim, self.attn_dropout) for _ in range(num_heads)])

        self.output_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, output_attention_probs=False):
        attention_outputs = [head_attention(x) for head_attention in self.multi_head_attention]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)

        attention_output = self.W_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attention_probs:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            return (attention_output, attention_probs)

class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: int=768,
                 mlp_size: int=3072,
                 dropout: float=0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.dropout = dropout

        self.layer_norm = nn.LayerNorm(normalized_shape=self.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim, out_features=self.mlp_size),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.mlp_size, out_features=self.embedding_dim),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):

    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 key_dim:int=768,
                 value_dim=None,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout
        self.attn_dropout = attn_dropout

        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dim=self.embedding_dim, num_heads=self.num_heads, key_dim=self.key_dim, value_dim=self.value_dim, attn_dropout=self.attn_dropout)
        self.mlp_block =  MLPBlock(embedding_dim=self.embedding_dim, mlp_size=self.mlp_size, dropout=self.mlp_dropout)

    def forward(self, x, output_attentions=False):

        attention_output, attention_probs = self.msa_block(x, output_attentions)

        x =  attention_output + x
        x = self.mlp_block(x) + x

        return (x, attention_probs)

class ViT(nn.Module):

    def __init__(self,
                 img_size: int=224,
                 in_channels: int=3,
                 patch_size: int=16,
                 num_transformer_layers: int=12,
                 embedding_dim: int=768,
                 mlp_size: int=3072,
                 num_heads: int=12,
                 value_dim: int=768,
                 key_dim: int=768,
                 attn_dropout: float=0,
                 mlp_dropout: float=0.1,
                 embedding_dropout: float=0.1,
                 num_classes: int=1000):
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        self.num_patches = (img_size * img_size) // patch_size**2
        self.img_size = img_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_transformer_layers = num_transformer_layers
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.key_dim = key_dim
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.embedding_dropout = embedding_dropout
        self.num_classes = num_classes

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, self.embedding_dim), requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=self.in_channels, patch_size=self.patch_size, embedding_dim=self.embedding_dim)

        self.transformer_encoder = nn.ModuleList([TransformerEncoderBlock(embedding_dim=self.embedding_dim,
                                                                            num_heads=self.num_heads,
                                                                            key_dim=self.key_dim,
                                                                            value_dim=self.value_dim,
                                                                            mlp_size=self.mlp_size,
                                                                            mlp_dropout=self.mlp_dropout,
                                                                            attn_dropout=self.attn_dropout) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, out_features=self.num_classes)
        )

    def forward(self, x, output_attentions=False):

        attention_probs = []

        x = self.patch_embedding(x)

        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)

        x = self.position_embedding + x
        x = self.embedding_dropout(x)

        for transformerEncoderBlock in self.transformer_encoder:
            x, probs = transformerEncoderBlock(x, output_attentions)
            if output_attentions:
                attention_probs.append(probs)

        logits = self.classifier(x[:, 0])

        if not output_attentions:
            return logits
        else:
            return (logits, attention_probs)