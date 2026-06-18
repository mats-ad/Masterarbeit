import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from utils.seed import get_device

class SPTPatchEmbed(nn.Module):
    """
    Shifted Patch Tokenization (SPT) as Patch Embedding replacement for ViT.
    Produces patch tokens with increased locality inductive bias by concatenating
    shifted versions of the input image along channel dimension.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        vanilla=False,
        use_ln=True,
        eps=1e-6
    ):
        super().__init__()
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
        self.half_patch = self.patch_size // 2
        self.vanilla = vanilla
        self.use_ln = use_ln and (not vanilla)
        
        effective_in_chans = in_chans if vanilla else in_chans * 5
        
        self.proj = nn.Conv2d(
            effective_in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        self.norm = nn.LayerNorm(embed_dim, eps=eps) if self.use_ln else nn.Identity()
        
    def _crop_shift_pad(self, x, mode: str):
        """
        Create diagonally shifted image by cropping and padding.
        x: (B, C, H, W)
        """
        hp = self.half_patch
        B, C, H, W = x.shape

        if mode == "left-up":
            top_crop, left_crop = hp, hp
            top_pad, left_pad = 0, 0
        elif mode == "left-down":
            top_crop, left_crop = 0, hp
            top_pad, left_pad = hp, 0
        elif mode == "right-up":
            top_crop, left_crop = hp, 0
            top_pad, left_pad = 0, hp
        elif mode == "right-down":
            top_crop, left_crop = 0, 0
            top_pad, left_pad = hp, hp
        else:
            raise ValueError(mode)

        x_crop = x[:, :, 
                top_crop:top_crop + (H - hp),
                left_crop:left_crop + (W - hp)]

        pad_bottom = H - x_crop.shape[2] - top_pad
        pad_right = W - x_crop.shape[3] - left_pad

        x_pad = F.pad(
            x_crop,
            (left_pad, pad_right, top_pad, pad_bottom)
        )

        return x_pad
    
    def forward(self, x):
        if not self.vanilla:
            x = torch.cat(
                [
                x, 
                self._crop_shift_pad(x, "left-up"),
                self._crop_shift_pad(x, "left-down"),
                self._crop_shift_pad(x, "right-up"),
                self._crop_shift_pad(x, "right-down"),
                ],
                dim = 1
            )
        
        x = self.proj(x)
        
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)
        
        return x
    
class LSA_Attention(nn.Module):
    """
    Drop-in replacement for timm ViT Attention with:
      - diagonal mask (no self-token attention)
      - learnable per-head scale (temperature)
    Based on common SPT+LSA implementations.  [oai_citation:1‡GitHub](https://raw.githubusercontent.com/aanna0701/SPT_LSA_ViT/main/models/vit.py)
    """
    def __init__(self, attn_module: nn.Module):
        super().__init__()
        
        # Grab needed parts from timm's Attention
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.attn_drop = attn_module.attn_drop
        self.proj = attn_module.proj
        self.proj_drop = attn_module.proj_drop
        
        # timm attention uses a single scale
        # LSA makes it learnable per head
        init_scale = getattr(attn_module, "scale", self.head_dim ** -0.5)
        self.scale = nn.Parameter(torch.ones(self.num_heads) * float(init_scale))
        
        # qkv projection
        self.qkv = attn_module.qkv
        
        # small cache for diagonal masks by token-length (optional speed)
        self._mask_cache = {}

    def _diag_mask(self, n: int, device):
        key = (n, device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        m = torch.eye(n, device=device, dtype=torch.bool)
        self._mask_cache[key] = m
        return m

    def forward(self, x, attn_mask=None, **kwargs):
        # attn_mask is passed by some timm blocks
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))
        attn = attn * self.scale.view(1, self.num_heads, 1, 1)

        # LSA: mask diagonal (no self-attention)
        diag = self._diag_mask(N, x.device)
        attn = attn.masked_fill(diag.view(1, 1, N, N), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
        
# Enable SPT when it is chosen in the search space by the nas
def enable_spt(model, img_size=224, patch_size=16, vanilla=False):
    """
    Replace model.patch_embed with SPTPatchEmbed while keeping everything else identical.
    Assumes ViT-Small (embed_dim=384).
    """
    embed_dim = getattr(model, "embed_dim", 384)
    in_chans = 3
    
    model.patch_embed = SPTPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        vanilla=vanilla,
        use_ln=True
    )
    
    return model

# Enable LSA when it is chosen in the search space by the nas
def enable_lsa(model):
    for blk in model.blocks:
        blk.attn = LSA_Attention(blk.attn)
    return model

def build_vit_from_cfg(num_classes, cfg):
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
    )

    if cfg["use_spt"] == 1:
        model = enable_spt(model, img_size=224, patch_size=16, vanilla=False)
        
    if cfg["use_lsa"] == 1: model = enable_lsa(model)
    

    mode = cfg.get("finetune_mode", "last2")

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Train head always
    for p in model.head.parameters():
        p.requires_grad = True

    # Finetune modes
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "last2":
        for blk in model.blocks[-2:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.norm.parameters():
            p.requires_grad = True
    elif mode == "last4":
        for blk in model.blocks[-4:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.norm.parameters():
            p.requires_grad = True
    elif mode == "head":
        pass
    else:
        raise ValueError(mode)

    return model.to(get_device())