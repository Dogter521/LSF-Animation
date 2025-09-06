import torch
from torch import nn
import pytorch_lightning as pl

from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding
from timm.models.vision_transformer import DropPath, Mlp

# -----------------------------------------------------------------------------
#  Cross‑modal Transformer Predictor with DenseAV Fusion Tokens
# -----------------------------------------------------------------------------

class CrossAttention_DenseAVInteractions(nn.Module):
    """Cartesian cross‑attention between audio (xa) and emotion (xv) streams, queried by fusion tokens."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dim_ratio=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = int(dim * dim_ratio)

        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(dim * 2, self.dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xmm, xa, xv):
        B, N_mm, _ = xmm.shape
        N_a, N_v = xa.size(1), xv.size(1)
        # Cartesian product concat (v,a) → (B, N_v*N_a, 2C)
        x_va = torch.cat([
            xv.unsqueeze(2).repeat(1, 1, N_a, 1),
            xa.unsqueeze(1).repeat(1, N_v, 1, 1)
        ], dim=3).flatten(1, 2)

        q = self.q(xmm).reshape(B, N_mm, self.num_heads, -1).permute(0, 2, 1, 3)
        k, v = self.kv(x_va).reshape(B, N_v*N_a, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N_mm, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FusionBlock_DenseAVInteractions(nn.Module):
    def __init__(self, dim, num_heads, attn_ratio=0.25, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_mm = norm_layer(dim)
        self.norm_a = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention_DenseAVInteractions(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, dim_ratio=attn_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, xmm, xv, xa):
        src_mm, src_v, src_a = self.norm_mm(xmm), self.norm_v(xv), self.norm_a(xa)
        out = self.attn(src_mm, xa=src_a, xv=src_v)
        xmm = xmm + self.drop_path(out)
        xmm = xmm + self.drop_path(self.mlp(self.norm_mlp(xmm)))
        return xmm


# -----------------------------------------------------------------------------
class TransformerPredictor(pl.LightningModule):
    """Two‑stream Transformer encoders (audio, emotion) with per‑layer DenseAV fusion tokens."""

    def __init__(self, latent_dim: int,
                 num_layers: int,
                 num_heads: int,
                 quant_factor: int,
                 intermediate_size: int,
                 audio_dim: int,
                 emotion_dim: int,
                 one_hot_dim: int,
                 n_fusion_tokens: int = 4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # ---------- projections ----------
        self.audio_feat = nn.Linear(audio_dim, latent_dim)
        self.emo_feat = nn.Linear(emotion_dim, latent_dim)
        self.style_proj = nn.Linear(one_hot_dim, latent_dim, bias=False)

        # squashers (same for both streams)
        def make_squasher():
            return nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm1d(latent_dim, affine=False))
        self.squash_a = make_squasher()
        self.squash_e = make_squasher()

        # embeddings
        self.lin_embed = LinearEmbedding(latent_dim, latent_dim)
        self.pos_enc = PositionalEncoding(latent_dim, batch_first=True)

        # one‑layer Transformer blocks stacked manually
        self.audio_blocks = nn.ModuleList([
            Transformer(latent_dim, latent_dim, 1, num_heads, intermediate_size)
            for _ in range(num_layers)])
        self.emo_blocks = nn.ModuleList([
            Transformer(latent_dim, latent_dim, 1, num_heads, intermediate_size)
            for _ in range(num_layers)])

        # fusion tokens + fusion blocks
        self.fusion_tokens = nn.Parameter(torch.randn(1, n_fusion_tokens, latent_dim) * 0.02)
        self.fusion_blocks = nn.ModuleList([
            FusionBlock_DenseAVInteractions(latent_dim, num_heads,
                                            mlp_ratio=intermediate_size/latent_dim)
            for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(latent_dim)

    # ------------------------------------------------------------------
    def forward(self, audio, emotion_feature, style_one_hot):
        """audio: B×T×audio_dim, emotion_feature: B×T×emotion_dim"""
        B = audio.size(0)
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}

        # 1) project and style scale
        xa = self.audio_feat(audio)
        xv = self.emo_feat(emotion_feature)
        style = self.style_proj(style_one_hot).unsqueeze(1)  # (B,1,C)
        xa *= style
        xv *= style

        # 2) squash conv (C,T)↔(T,C)
        xa = self.squash_a(xa.permute(0,2,1)).permute(0,2,1)
        xv = self.squash_e(xv.permute(0,2,1)).permute(0,2,1)

        # 3) embed + pos
        xa = self.pos_enc(self.lin_embed(xa))
        xv = self.pos_enc(self.lin_embed(xv))

        # 4) fusion per layer
        x_mm = self.fusion_tokens.expand(B, -1, -1)
        nF = x_mm.shape[1]
        for blk_a, blk_v, blk_f in zip(self.audio_blocks, self.emo_blocks, self.fusion_blocks):
            nA, nV = xa.size(1), xv.size(1)
            # prepend fusion tokens
            xa_cat = torch.cat([x_mm, xa], dim=1)
            xv_cat = torch.cat([x_mm, xv], dim=1)

            xa_cat = blk_a((xa_cat, dummy_mask))
            xv_cat = blk_v((xv_cat, dummy_mask))
            # split back
            _, xa = xa_cat.split((nF, nA), dim=1)
            _, xv = xv_cat.split((nF, nV), dim=1)

            # update fusion tokens
            x_mm = blk_f(x_mm, xv, xa)

        return xa
