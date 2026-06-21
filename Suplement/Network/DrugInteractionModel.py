import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Suplement.Network.Cap_net import MarginLoss
from Suplement.Network.sccaps_head import SCCapsNetHead


class CrossAttention(nn.Module):
    def __init__(self, view_dims, hidden_dim=64):
        super().__init__()
        self.view_names = list(view_dims.keys())
        self.hidden_dim = hidden_dim

        self.query_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim)
            for name in view_dims.keys()
        })
        self.key_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim)
            for name in view_dims.keys()
        })
        self.value_proj = nn.ModuleDict({
            name: nn.Linear(hidden_dim, hidden_dim)
            for name in view_dims.keys()
        })

    def forward(self, drugA_views, drugB_views):
        """
        drugA_views / drugB_views: dict[name] -> [B, H]
        return fused_views[name]: [B, H]
        """
        fused_views = {name: [] for name in self.view_names}

        for name_A in self.view_names:
            for name_B in self.view_names:
                Q = self.query_proj[name_A](drugA_views[name_A])  # [B, H]
                K = self.key_proj[name_B](drugB_views[name_B])    # [B, H]
                V = self.value_proj[name_B](drugB_views[name_B])  # [B, H]

                attn_scores = torch.matmul(
                    Q, K.transpose(-2, -1)
                ) / math.sqrt(Q.size(-1))         # [B, B]
                attn_weights = F.softmax(attn_scores, dim=-1)
                fused = torch.matmul(attn_weights, V)             # [B, H]

                fused_views[name_A].append(fused)

        final_views = {}
        for name in self.view_names:
            # [B, num_views, H] -> mean over views
            all_fusions = torch.stack(fused_views[name], dim=1)
            final_views[name] = torch.mean(all_fusions, dim=1)

        return final_views


class DrugInteractionModel(nn.Module):

    def __init__(self, feature_config, hidden_dim=64, num_classes=2, temperature=0.5):
        super().__init__()
        self.feature_config = feature_config
        self.view_dims = {k: v['dim'] for k, v in feature_config.items()}

        self.bert_cnn = None
        if 'bert' in self.view_dims:
            self.bert_cnn = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(8, 4, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(4, 1, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        self.view_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(cfg['dim'], hidden_dim),
                nn.ReLU()
            )
            for name, cfg in feature_config.items()
        })

        self.view_projections = nn.ModuleDict({
            name: nn.Linear(cfg['dim'], 64)
            for name, cfg in feature_config.items()
        })

        # Cross-view Cross-attention
        self.cross_attention = CrossAttention(
            view_dims=self.view_dims,
            hidden_dim=hidden_dim
        )

        # Spiking Conv Capsule Head
        self.capsnet = SCCapsNetHead(
            input_dim=hidden_dim * len(self.view_dims),
            num_classes=num_classes,
            T=10,
            num_primary=8,
            primary_dim=8,
            digit_dim=16,
            use_fc_head=True,
            use_sconv=True,   
            sconv_layers=3,
            sconv_T=4,
            lif_tau=2.0,
        )

        self.margin_loss = MarginLoss(m_plus=0.9, m_minus=0.1, lambda_=0.5)
        self.sym_proj = None  

    # -------- utilities -------- #
    def split_features(self, x):
        """
        x: [B, 1, total_dim]
        """
        return {
            name: x[:, :, cfg['start']: cfg['start'] + cfg['dim']]
            for name, cfg in self.feature_config.items()
        }

    def _preprocess_view(self, name, view_tensor):
        """Apply optional preprocessing (e.g., CNN) for specific feature views."""
        if name == 'bert' and self.bert_cnn is not None:
            # view_tensor: [B, 1, dim] -> CNN -> [B, 1, dim] -> flatten to [B, dim]
            return self.bert_cnn(view_tensor).squeeze(1)

        return view_tensor.squeeze(1)

    # -------- public forward -------- #
    def forward(self, drug1, drug2, return_logits=False):
        probs, v, logits = self._forward(drug1, drug2)
        return logits if return_logits else probs

    # -------- internal forward -------- #
    def _forward(self, drug1, drug2):
        """
        drug1, drug2: [B, total_feature_dim]
        """
        drug1_views = self.split_features(drug1.unsqueeze(1))  # dict[name]: [B,1,dim]
        drug2_views = self.split_features(drug2.unsqueeze(1))

        encoded_A = {
            k: self.view_encoders[k](self._preprocess_view(k, v))
            for k, v in drug1_views.items()
        }  # [B, H]
        encoded_B = {
            k: self.view_encoders[k](self._preprocess_view(k, v))
            for k, v in drug2_views.items()
        }

        projected_A = {
            k: self.view_projections[k](self._preprocess_view(k, v))
            for k, v in drug1_views.items()
        }  # [B, 64]
        projected_B = {
            k: self.view_projections[k](self._preprocess_view(k, v))
            for k, v in drug2_views.items()
        }

        fused_A = self.cross_attention(encoded_A, encoded_B)  # dict[name]: [B, H]
        fused_B = self.cross_attention(encoded_B, encoded_A)

        view_names = list(self.view_dims.keys())

        combined_A = torch.cat([fused_A[n] for n in view_names], dim=1)  # [B, H * V]
        combined_B = torch.cat([fused_B[n] for n in view_names], dim=1)

        projected_A_combined = torch.cat([projected_A[n] for n in view_names], dim=1)
        projected_B_combined = torch.cat([projected_B[n] for n in view_names], dim=1)

        final_A = (combined_A + projected_A_combined) / 2
        final_B = (combined_B + projected_B_combined) / 2

        sym = torch.cat(
            [final_A, final_B, torch.abs(final_A - final_B), final_A * final_B],
            dim=1,
        )  # [B, 4 * H * V]

        if self.sym_proj is None:
            self.sym_proj = nn.Sequential(
                nn.Linear(sym.size(1), final_A.size(1)),
                nn.ReLU()
            ).to(sym.device)

        combined = self.sym_proj(sym)  # [B, hidden_dim * num_views]

        probs, digit_caps, logits = self.capsnet(combined)
        return probs, digit_caps, logits
