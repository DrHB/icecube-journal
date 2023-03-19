import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

#BEiTv2 block
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        if self.gamma_1 is None:
            xn = self.norm1(x)
            x = x + self.drop_path(self.attn(xn,xn,xn,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0])
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            xn = self.norm1(x)
            x = x + self.drop_path(self.gamma_1 * self.attn(xn,xn,xn,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0])
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Extractor(nn.Module):
    def __init__(self, dim_base=128, dim=384):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim=dim_base)
        #self.emb2 = SinusoidalPosEmb(dim=dim_base//2)
        self.aux_emb = nn.Embedding(2,dim_base//2)
        #self.qe_emb = nn.Embedding(2,dim_base//2)
        self.proj = nn.Linear(11*dim_base//2,dim)
        
    def forward(self, x, Lmax=None):
        pos = x['pos'] if Lmax is None else x['pos'][:,:Lmax]
        charge = x['charge'] if Lmax is None else x['charge'][:,:Lmax]
        time = x['time'] if Lmax is None else x['time'][:,:Lmax]
        auxiliary = x['auxiliary'] if Lmax is None else x['auxiliary'][:,:Lmax]
        qe = x['qe'] if Lmax is None else x['qe'][:,:Lmax]
        ice_properties = x['ice_properties'] if Lmax is None else x['ice_properties'][:,:Lmax]
        
        x = torch.cat([self.emb(100*pos).flatten(-2), self.emb(40*charge),
                       self.emb(100*time),self.aux_emb(auxiliary)#,self.qe_emb(qe),
                       #self.emb2(50*ice_properties).flatten(-2)
                      ],-1)
        x = self.proj(x)
        return x

class DeepIceModel(nn.Module):
    def __init__(self, dim=384, dim_base=128, depth=12, use_checkpoint=False, **kwargs):
        super().__init__()
        self.extractor = Extractor(dim_base,dim)
        self.cls_token = nn.Linear(dim,1,bias=False)
        self.blocks = nn.ModuleList([ 
            Block(
                dim=dim, num_heads=dim//64, mlp_ratio=4, drop_path=0.0*(i/(depth-1)), init_values=1,)
            for i in range(depth)])
        #self.blocks = nn.ModuleList([ 
        #    nn.TransformerEncoderLayer(dim,dim//64,dim*4,dropout=0,
        #        activation=nn.GELU(), batch_first=True, norm_first=True)
        #    for i in range(depth)])

        self.proj_out = nn.Linear(dim,3)
        self.use_checkpoint = use_checkpoint
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=.02)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}
    
    def forward(self, x0):
        mask = x0['mask']
        B,_ = mask.shape
        mask = torch.cat([torch.ones(B,1,dtype=mask.dtype, device=mask.device),mask],1)
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        Lmax = mask.sum(-1).max()
        
        x = self.extractor(x0, Lmax-1)
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B,-1,-1)
        x = torch.cat([cls_token,x],1)
        x,mask,attn_mask = x[:,:Lmax], mask[:,:Lmax], attn_mask[:,:Lmax]
        
        #Lmax = mask.max(0)[0].sum()
        Lmax = mask.sum(-1).max()
        x,mask,attn_mask = x[:,:Lmax], mask[:,:Lmax], attn_mask[:,:Lmax]
        #print(Lmax)
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else: x = blk(x, None, attn_mask)
                
        x = self.proj_out(x[:,0]) #cls token
        return x
        
    def get_layer_groups(self):
        def flatten_model(m):
            return [m] if not hasattr(m,'children') or len(list(m.children())) == 0 else \
                sum(map(flatten_model, list(m.children())), [])
        return [flatten_model(self)]

