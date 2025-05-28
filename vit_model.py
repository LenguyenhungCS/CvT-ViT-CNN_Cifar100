import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    def get_layer_weights(self):
        return torch.clone(self.fn.get_attn_weights())

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attn_weights = 0

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        self.attn_weights = torch.clone(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def get_attn_weights(self):
        return torch.clone(self.attn_weights)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    def get_attn_weights(self):
        attn_weights = []
        for attn, ff in self.layers:
            attn_weights.append(attn.get_layer_weights())
        attn_weights = torch.stack(attn_weights).squeeze(1)
        return attn_weights

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def get_attn_weights(self):
        return torch.clone(self.transformer.get_attn_weights())

def vit_b_cifar100(num_classes=100, image_size=32, patch_size=4, channels=3,
                     dropout=0.1, emb_dropout=0.1):
    """
    Tạo mô hình ViT-Base được cấu hình cho CIFAR-100.
    Patch size được chọn là 4 để phù hợp với kích thước ảnh 32x32.
    (32/4)^2 = 8^2 = 64 patches.
    """
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        channels=channels,
        dropout=dropout,
        emb_dropout=emb_dropout,
        pool='cls' # Sử dụng CLS token pooling
    )
    return model

if __name__ == '__main__':
    # Ví dụ cách sử dụng
    img_size = 32
    patch_s = 4
    n_classes = 100

    # Tạo mô hình ViT-B cho CIFAR-100
    vit_b_model = vit_b_cifar100(num_classes=n_classes, image_size=img_size, patch_size=patch_s)

    # Tạo một tensor ảnh giả lập (batch_size, channels, height, width)
    dummy_img = torch.randn(2, 3, img_size, img_size)

    # Đưa ảnh qua mô hình
    preds = vit_b_model(dummy_img)

    print(f"Kích thước output của mô hình: {preds.shape}") # torch.Size([2, 100])

    # Lấy trọng số attention (ví dụ)
    # attn_weights = vit_b_model.get_attn_weights()
    # print(f"Kích thước attention weights: {attn_weights.shape}")
    # attn_weights có kích thước (depth, batch_size, num_heads, num_patches+1, num_patches+1)
    # ví dụ: torch.Size([12, 2, 12, 65, 65]) cho cấu hình trên
    # Squeeze(1) trong Transformer.get_attn_weights() có thể không cần thiết nếu batch_size > 1 khi gọi get_layer_weights
    # Tuy nhiên, hiện tại get_layer_weights() trả về clone của self.attn_weights được tính trong forward cuối cùng.
    # Nếu cần attention của từng layer riêng lẻ và cho batch, cần điều chỉnh cách lấy attention.
    # Hiện tại, get_attn_weights sẽ trả về attention của batch cuối cùng đi qua model.
    # Để lấy attention weights đúng cách trong quá trình training hoặc inference cho một batch cụ thể,
    # bạn có thể cần gọi get_attn_weights() ngay sau khi gọi forward() cho batch đó,
    # hoặc điều chỉnh logic lưu trữ attention trong các lớp Attention và Transformer.

    # Với cấu trúc hiện tại, để lấy attention của batch dummy_img:
    _ = vit_b_model(dummy_img) # Chạy forward pass để tính attention cho batch này
    attn_weights_example = vit_b_model.get_attn_weights()
    print(f"Kích thước attention weights ví dụ (batch_size=2): {attn_weights_example.shape}")
    # Expected: (depth, batch_size, heads, num_patches+1, num_patches+1) -> (12, 2, 12, 65, 65)
    # vì squeeze(1) không ảnh hưởng nếu batch_size > 1.

    print("\\nVí dụ về cách lấy attention weights với batch_size=1:")
    vit_b_model_single_batch = vit_b_cifar100(num_classes=n_classes, image_size=img_size, patch_size=patch_s)
    dummy_img_single = torch.randn(1, 3, img_size, img_size)
    preds_single = vit_b_model_single_batch(dummy_img_single)
    attn_single = vit_b_model_single_batch.get_attn_weights()
    print(f"Shape attention cho batch_size=1: {attn_single.shape}")
    # Expected: (depth, heads, num_patches+1, num_patches+1) -> (12, 12, 65, 65)
    # do squeeze(1) sẽ loại bỏ chiều batch_size khi nó bằng 1.

    # Giải thích ngắn gọn về squeeze(1) trong get_attn_weights của Transformer:
    # 1. self.attn_weights trong lớp Attention có shape (B, H, N+1, N+1)
    # 2. attn.get_layer_weights() trả về clone của self.attn_weights.
    # 3. torch.stack(attn_weights) trong Transformer.get_attn_weights() tạo ra tensor shape (D, B, H, N+1, N+1).
    # 4. .squeeze(1) sẽ:
    #    - Nếu B=1 (batch_size=1), loại bỏ chiều thứ 1, kết quả là (D, H, N+1, N+1).
    #    - Nếu B>1, không làm gì cả (vì chiều thứ 1 không phải là 1), kết quả là (D, B, H, N+1, N+1).
    # Điều này giải thích tại sao shape output của get_attn_weights() khác nhau giữa batch_size=1 và batch_size>1. 