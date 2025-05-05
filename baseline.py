import torch
import torch.nn as nn
import torch.nn.functional as F

class QFormer(nn.Module):
    def __init__(self, 
                 hidden_dim=768,
                 num_queries=16,
                 num_layers=6,
                 num_heads=12,
                 image_feat_dim=320,
                 text_feat_dim=2048,
                 add_modality_embedding=True):
        super(QFormer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.add_modality_embedding = add_modality_embedding

        # Learnable query tokens: [1, num_queries, D]
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim))

        # Modality type embeddings (0=image, 1=text)
        if add_modality_embedding:
            self.modality_embed = nn.Embedding(2, hidden_dim)
        self.image_proj = nn.Linear(image_feat_dim, hidden_dim)
        self.text_proj = nn.Linear(text_feat_dim, hidden_dim)
        # Transformer encoder: Q interacts with K/V (image + text)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, image_feat, text_feat):
        """
        image_feat: [B, T_img, D]
        text_feat:  [B, T_txt, D]
        """
        
        B = image_feat.size(0)
        print(image_feat.shape,text_feat.shape)
        image_feat = self.image_proj(image_feat)
        text_feat = self.text_proj(text_feat)
        # Concatenate image + text features as K/V
        kv = torch.cat([image_feat, text_feat], dim=1)  # [B, T_img + T_txt, D]

        # Add modality type embedding
        if self.add_modality_embedding:
            T_img = image_feat.size(1)
            T_txt = text_feat.size(1)
            modality_ids = torch.cat([
                torch.zeros(T_img, dtype=torch.long),
                torch.ones(T_txt, dtype=torch.long)
            ], dim=0).to(image_feat.device)  # [T_img + T_txt]
            modality_embed = self.modality_embed(modality_ids)  # [T_img + T_txt, D]
            kv = kv + modality_embed.unsqueeze(0)  # broadcast to [B, T, D]

        # Expand learnable query tokens to batch size
        queries = self.query_tokens.expand(B, -1, -1)  # [B, N_query, D]

        # Q-Former: let queries attend to K/V
        # Transformer requires concat(Q, K/V)
        input_seq = torch.cat([queries, kv], dim=1)  # [B, N_query + T, D]
        output = self.transformer(input_seq)  # [B, N_query + T, D]

        # Return only the updated query tokens
        return output[:, :self.num_queries, :]  # [B, N_query, D]
    

class MLP(nn.Module):
    def __init__(self, image_dim=320, text_dim=2048, fused_dim=768,num_header =16):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.num_header = num_header
        self.mlp = nn.Sequential(
            nn.Linear(2 * fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim,fused_dim*16)
        )
        self.fused_dim=fused_dim
    def forward(self, image_feat, text_feat):
        """
        image_feat: [B, T_img, image_dim]
        text_feat: [B, T_txt, text_dim]
        """
        # 简单池化（mean pooling）
        image_repr = image_feat.mean(dim=1)  # [B, image_dim]
        text_repr = text_feat.mean(dim=1)    # [B, text_dim]

        # 映射到相同维度
        image_proj = self.image_proj(image_repr)  # [B, fused_dim]
        text_proj = self.text_proj(text_repr)     # [B, fused_dim]

        # 拼接后送入 MLP
        fused = torch.cat([image_proj, text_proj], dim=-1)  # [B, 2*fused_dim]
        
        output = self.mlp(fused).reshape(-1,self.num_header,self.fused_dim)  # [B, fused_dim]
        return output
    
# 第4种 gated-attention


class GatedAttentionFusion(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        # 融合门控（产生 alpha 权重）
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 alpha ∈ [0, 1]
        )

    def forward(self, img_feat, txt_feat):
        """
        img_feat: [B, D]
        txt_feat: [B, D]
        """
        fused_input = torch.cat([img_feat, txt_feat], dim=-1)  # [B, 2D]
        alpha = self.gate_mlp(fused_input)  # [B, 1]

        # 融合：注意这里 alpha 需要 broadcast
        fused = alpha * img_feat + (1 - alpha) * txt_feat  # [B, D]
        return fused

class AttentionFusionWrapper(nn.Module):
    def __init__(self, image_dim=320, text_dim=2048, fused_dim=768,num_header=16):
        super().__init__()
        self.img_proj = nn.Linear(image_dim, fused_dim)
        self.txt_proj = nn.Linear(text_dim, fused_dim)
        self.fusion = GatedAttentionFusion(input_dim=fused_dim)
        self.num_header = num_header
        self.fused_dim = fused_dim
        self.dim_transfer = nn.Linear(fused_dim,fused_dim*self.num_header)
    def forward(self, image_feat, text_feat):
        """
        image_feat: [B, T_img, 320]
        text_feat:  [B, T_txt, 2048]
        """
        # Mean pooling
        img_global = image_feat.mean(dim=1)  # [B, 320]
        txt_global = text_feat.mean(dim=1)  # [B, 2048]

        # Linear projection
        img_proj = self.img_proj(img_global)  # [B, 768]
        txt_proj = self.txt_proj(txt_global)  # [B, 768]

        # Gated fusion
        fused = self.fusion(img_proj, txt_proj)  # [B, 768]
        fused = self.dim_transfer(fused).reshape(-1,self.num_header,self.fused_dim)
        return fused