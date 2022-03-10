import torch
from torch import nn
from torch.nn import functional as F
from .vit import ViT, ConvViTModel
from .u_transformer import Encoder as UEncoder

class ConvViUT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, vit_depth=2, max_hops=6, vit_conv_strides=None, vit_conv_channels=None, act=False, equivariant=False,
                 vit_dim=512, vit_heads=16, u_heads=8, mlp_dim=1024, internal_enc_layers=1, multiloss=False, pretrained=None, dropout=0.5, uncertainty_weighting="task-dependent"):
        super().__init__()
        self.vit = ViT(
            image_size=128,
            patch_size=16,
            num_classes=-1,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=mlp_dim,
            dropout=0.1,
            emb_dropout=0.1,
            use_conv=True,
            conv_strides=vit_conv_strides,
            conv_channels=vit_conv_channels,
            equivariant=equivariant,
            pretrained=pretrained
        )

        print('Vit Depth: {}; U Transf Depth: {}'.format(vit_depth, max_hops))

        self.max_hops = max_hops
        if self.max_hops == 0:
            self.u_transformer = None
        else:
            self.u_transformer = UEncoder(embedding_size=vit_dim, hidden_size=vit_dim, num_layers=max_hops, num_heads=u_heads,
                                      total_key_depth=mlp_dim, total_value_depth=mlp_dim, filter_size=mlp_dim, max_length=(img_size // patch_size)**2 + 1,
                                      act=act, internal_enc_layers=internal_enc_layers)
        self.to_latent = nn.Identity()
        self.cls_head = nn.Linear(vit_dim, 3 if uncertainty_weighting == 'data-dependent' else 2)
        self.dropout = nn.Dropout(dropout)
        self.multiloss = multiloss
        self.loss = nn.CrossEntropyLoss()
        self.uncertainty_weighting = uncertainty_weighting
        if uncertainty_weighting == 'task-dependent':
            self.uncertainty_loss_weights = nn.Parameter(torch.ones(max_hops) * -2.3)

    def forward(self, img, targets):
        x = self.vit(img)   # B x S x dim
        if self.u_transformer is not None:
            x, aux = self.u_transformer(x, intermediate_outputs=self.multiloss)   # B x S x dim
            if aux is not None:
                _, n_updates = aux
                n_updates = float(n_updates.mean().item())
            else:
                n_updates = self.max_hops
        if self.multiloss:
            # raise NotImplementedError
            # x is B x num_hops x S x dim
            out = x[:, :, 0, :] # take the CLS from every computational hop  # B x num_hops x dim
            out = self.to_latent(out)
            out = self.dropout(out)
            out = self.cls_head(out)
            loss = self.compute_loss(out, targets)
            # take the most likely prediction
            out = out[..., :2]
            probs = torch.softmax(out, dim=2)
            higher_probs_idx = torch.argmax(torch.abs(probs[:, :, 0] - 0.5), dim=1)
            out = [out[b, i, :] for b, i in enumerate(higher_probs_idx)]
            out = torch.stack(out)
            return loss, out, float(higher_probs_idx.float().mean())

        else:
            out = x[:, 0, :]
            out = self.to_latent(out)
            out = self.dropout(out)
            out = self.cls_head(out)
            # out = out[:, :2]    #TODO: to remove! (in case we erroneously left 5 classes)
            loss = self.loss(out, targets)
            return loss, out, 0

    def compute_loss(self, out, targets):
        if self.multiloss:
            bs, seq_length = out.shape[:2]
            if self.uncertainty_weighting == 'data-dependent':
                out = out.view(-1, out.shape[2])
                out, log_variance = out[:, :2], out[:, 2]
                targets = targets.unsqueeze(1).expand(-1, seq_length).flatten()    # the target is the same at every timestep
                bce_loss = F.cross_entropy(out, targets, reduction='none')
                final_loss = (bce_loss * torch.exp(-log_variance) + log_variance) * 0.5
                final_loss = final_loss.mean()
                return final_loss
            elif self.uncertainty_weighting == 'task-dependent':
                losses = [self.loss(out[:, i, :], targets) for i in range(out.shape[1])]
                if self.uncertainty_loss_weights is not None:
                    losses = [(l * torch.exp(-w) + w) * 0.5 for l, w in zip(losses, self.uncertainty_loss_weights)]
                return sum(losses) / len(losses)
            elif self.uncertainty_weighting == 'mean':
                targets = targets.unsqueeze(1).expand(-1, seq_length)  # the target is the same at every timestep
                return self.loss(out.view(-1, 2), targets.flatten())
            else:
                raise ValueError('Uncertainty weighting mode not recognized!')
        else:
            return self.loss(out, targets)

    def get_last_selfattention(self, img):
        x = self.vit(img)
        attn = self.u_transformer.get_last_selfattention(x)
        return attn

