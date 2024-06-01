import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_swinir import *
from .blocks import *
from einops import rearrange

from basicsr.utils.registry import ARCH_REGISTRY
import pdb
##########################################################################
##---------- CTBUnet -----------------------
@ARCH_REGISTRY.register()
class CTBUnet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """
    def __init__(self, img_size=96, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[2, 2, 2], depths_decoder=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, token_size=1, img_range=1.,
                 use_checkpoint=False, **kwargs):
        super(CTBUnet, self).__init__()
        
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        
        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.head = conv(in_chans, embed_dim, 3)
        self.linear_encoding = nn.Linear(embed_dim, embed_dim)
        
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)    
        self.num_layers_decoder = len(depths_decoder)
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]  # stochastic depth decay rule
        
        # build transformer encoder layers
        self.swin_layers = nn.ModuleList()
        self.swin_downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            #pdb.set_trace()
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
            self.swin_layers.append(layer)
            swin_downsample_layer = PatchMerging((patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)), 
                                    dim=int(embed_dim * 2 ** i_layer), norm_layer=nn.LayerNorm)
            self.swin_downsample_layers.append(swin_downsample_layer)
        
        # build CNN encoder layers
        self.CNN_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            #pdb.set_trace()
            layer = ResidualBlock(int(embed_dim * 2 ** i_layer), int(embed_dim * 2 ** i_layer))
            self.CNN_layers.append(layer)
            downsample_layer = ConvBlock(int(embed_dim * 2 ** i_layer), int((embed_dim * 2 ** i_layer) * 2), 3, 'down')
            self.downsample_layers.append(downsample_layer)
        
        # build transformer decoder layers
        self.swin_layers_decoder = nn.ModuleList()
        self.swin_upsample_layers = nn.ModuleList()
        self.swin_concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers_decoder):
            #pdb.set_trace()
            layer_decode = BasicLayer(dim=int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)),
                               input_resolution=(patches_resolution[0] // (2 ** (self.num_layers_decoder-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers_decoder-1-i_layer))),
                               depth=depths_decoder[i_layer],
                               num_heads=num_heads[(self.num_layers_decoder-1-i_layer)],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths_decoder[:(self.num_layers_decoder-1-i_layer)]):sum(depths_decoder[:(self.num_layers_decoder-1-i_layer) + 1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)            
            self.swin_layers_decoder.append(layer_decode)
            if i_layer < self.num_layers_decoder - 1:
                upsample_layer = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers_decoder-1-i_layer)),
                    patches_resolution[1] // (2 ** (self.num_layers_decoder-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
                self.swin_upsample_layers.append(upsample_layer)
                
                concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers_decoder-2-i_layer)),
                int(embed_dim*2**(self.num_layers_decoder-2-i_layer)))
                self.swin_concat_back_dim.append(concat_linear)
                
        # build CNN decoder layers
        self.CNN_layers_decoder = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers_decoder):
            layer_decode = ResidualBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)))
            self.CNN_layers_decoder.append(layer_decode)
            
            if i_layer < self.num_layers_decoder - 1:
                #pdb.set_trace()
                upsample_layer = ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)/2), 3, 'up')
                self.upsample_layers.append(upsample_layer)
            
                concat_conv = nn.Conv2d(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)/2), kernel_size=1, stride=1, padding=0, bias=True)
                self.concat_back_dim.append(concat_conv)
        
        #####################################################################################################
        # build bilateral fusion layers
        # CNN Fusion Blocks
        self.channel_fusion_block = nn.ModuleList()
        self.spatial_fusion_block = nn.ModuleList()
        self.fusion_CNN = nn.ModuleList()
        self.fusion_Transformer = nn.ModuleList()
        self.fusion_patch_unembed = nn.ModuleList()
        self.fusion_patch_embed = nn.ModuleList()
        self.token_size = token_size
        #pdb.set_trace()
        for i_layer in range(self.num_layers):
            #pdb.set_trace()
            self.channel_fusion_block.append(
            nn.Sequential(
                ConvBlock(int(embed_dim * 2 ** i_layer * 2), int(embed_dim * 2 ** i_layer * 2), 1),
                ConvBlock(int(embed_dim * 2 ** i_layer * 2), int(embed_dim * 2 ** i_layer * 2), 1),
                ConvBlock(int(embed_dim * 2 ** i_layer * 2), int(embed_dim * 2 ** i_layer * 2), 1),
                ConvBlock(int(embed_dim * 2 ** i_layer * 2), int(embed_dim * 2 ** i_layer * 2), 1),
            ))
            hidden_dim = int(embed_dim * 2 ** i_layer * mlp_ratio) * (self.token_size**2)
            dim_head = int(embed_dim * 2 ** i_layer) * (self.token_size**2) // num_heads[i_layer]
            dropout_rate = 0
            self.spatial_fusion_block.append(
                nn.ModuleList([
                # Two cross-attentions
                PreNorm2(
                    int(embed_dim * 2 ** i_layer)*(self.token_size**2),
                    CrossAttention(int(embed_dim * 2 ** i_layer)*(self.token_size**2), num_heads[i_layer], dim_head, dropout_rate)
                ),
                PreNorm2(
                    int(embed_dim * 2 ** i_layer)*(self.token_size**2),
                    CrossAttention(int(embed_dim * 2 ** i_layer)*(self.token_size**2), num_heads[i_layer], dim_head, dropout_rate)
                ),
                # conventional FFN after the attention
                PreNorm(
                    int(embed_dim * 2 ** i_layer)*(self.token_size**2),
                    FeedForward(int(embed_dim * 2 ** i_layer)*(self.token_size**2), hidden_dim, dropout_rate)
                ),
                PreNorm(
                    int(embed_dim * 2 ** i_layer)*(self.token_size**2),
                    FeedForward(int(embed_dim * 2 ** i_layer)*(self.token_size**2), hidden_dim, dropout_rate)
                ),
                ]))
            #pdb.set_trace()
            self.fusion_CNN.append(ConvBlock(int(embed_dim * 2 ** i_layer * 2), int(embed_dim * 2 ** i_layer), 1, skip=False))
            self.fusion_Transformer.append(ConvBlock(int(embed_dim * 2 ** i_layer * 2), int(embed_dim * 2 ** i_layer), 1, skip=False))
            self.fusion_patch_unembed.append(PatchUnEmbed(img_size=int(img_size / (2 ** i_layer)), patch_size=patch_size, in_chans=int(embed_dim * 2 ** i_layer), embed_dim=int(embed_dim * 2 ** i_layer),
            norm_layer=norm_layer if self.patch_norm else None))
            self.fusion_patch_embed.append(PatchEmbed(img_size=int(img_size / (2 ** i_layer)), patch_size=patch_size, in_chans=int(embed_dim * 2 ** i_layer), embed_dim=int(embed_dim * 2 ** i_layer),
            norm_layer=norm_layer if self.patch_norm else None))
            
        
        for i_layer in range(self.num_layers_decoder):
            #pdb.set_trace()
            self.channel_fusion_block.append(
            nn.Sequential(
                ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), 1),
                ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), 1),
                ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), 1),
                ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), 1),
            ))
            hidden_dim = int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * mlp_ratio) * (self.token_size**2)
            dim_head = int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)) * (self.token_size**2) // num_heads[(self.num_layers_decoder-1-i_layer)]
            dropout_rate = 0
            self.spatial_fusion_block.append(
                nn.ModuleList([
                # Two cross-attentions
                PreNorm2(
                    int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2),
                    CrossAttention(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2), num_heads[(self.num_layers_decoder-1-i_layer)], dim_head, dropout_rate)
                ),
                PreNorm2(
                    int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2),
                    CrossAttention(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2), num_heads[(self.num_layers_decoder-1-i_layer)], dim_head, dropout_rate)
                ),
                # conventional FFN after the attention
                PreNorm(
                    int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2),
                    FeedForward(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2), hidden_dim, dropout_rate)
                ),
                PreNorm(
                    int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2),
                    FeedForward(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer))*(self.token_size**2), hidden_dim, dropout_rate)
                ),
                ]))    
            #pdb.set_trace() 
            self.fusion_CNN.append(ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), 1, skip=False))
            self.fusion_Transformer.append(ConvBlock(int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer) * 2), int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), 1, skip=False))
            self.fusion_patch_unembed.append(PatchUnEmbed(img_size=img_size / (2 ** (self.num_layers_decoder-1-i_layer)), patch_size=patch_size, in_chans=int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), embed_dim=int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)),
            norm_layer=norm_layer if self.patch_norm else None))
            self.fusion_patch_embed.append(PatchEmbed(img_size=img_size / (2 ** (self.num_layers_decoder-1-i_layer)), patch_size=patch_size, in_chans=int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)), embed_dim=int(embed_dim * 2 ** (self.num_layers_decoder-1-i_layer)),
            norm_layer=norm_layer if self.patch_norm else None))
        
        self.norm = norm_layer(int(embed_dim * 2 ** (self.num_layers)))
        self.norm_up= norm_layer(embed_dim)
        
        #####################################################################################################
        #build attention gate
        self.pred_patch_unembed = PatchUnEmbed(
            img_size=img_size / (int(embed_dim * 2 ** (self.num_layers-1))), patch_size=patch_size, in_chans=int(embed_dim * 2 ** (self.num_layers-1)), embed_dim=int(embed_dim * 2 ** (self.num_layers-1)),
            norm_layer=norm_layer if self.patch_norm else None)
        self.conv_head = conv(int(embed_dim * 2 ** self.num_layers), int(embed_dim * 2 ** (self.num_layers - 1)), 3)
        self.pred_head = conv(int(embed_dim * 2 ** (self.num_layers-1)), 1, 3)
        self.attentions = nn.Sequential(
                          nn.Conv2d(1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                          nn.Sigmoid())
                          
        self.attention_CNN = nn.ModuleList()
        self.attention_Transformer = nn.ModuleList()
        self.attention_patch_unembed = nn.ModuleList()
        self.attention_patch_embed = nn.ModuleList()
        #pdb.set_trace()
        for i_layer in range(self.num_layers_decoder - 1):
            self.attention_patch_unembed.append(PatchUnEmbed(img_size=img_size / (2 ** (self.num_layers_decoder-2-i_layer)), patch_size=patch_size, in_chans=int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), embed_dim=int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), norm_layer=norm_layer if self.patch_norm else None))
            self.attention_patch_embed.append(PatchEmbed(img_size=img_size / (2 ** (self.num_layers_decoder-2-i_layer)), patch_size=patch_size, in_chans=int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), embed_dim=int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), norm_layer=norm_layer if self.patch_norm else None))
            self.attention_CNN.append(nn.ModuleList([
                nn.Conv2d(int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), 1, 1),
                nn.Conv2d(int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), 1, 1),
            ]))
            self.attention_Transformer.append(nn.ModuleList([
                nn.Conv2d(int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), 1, 1),
                nn.Conv2d(int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), int(embed_dim * 2 ** (self.num_layers_decoder-2-i_layer)), 1, 1),
            ]))
        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        self.conv_last = conv(embed_dim * 2, embed_dim, 3)
        self.tail = conv(embed_dim, 3, 3)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #attention gate
    def forward_attention_transformer(self, i, x, mask, x_size):
        #pdb.set_trace()
        x = self.attention_patch_unembed[i](x, x_size)
        x_size = (x.shape[2], x.shape[3])
        mask = F.interpolate(mask, size=x_size, mode='bilinear', align_corners=False)
        out = torch.mul(x, mask)
        out = self.attention_Transformer[i][0](out)
        x = self.attention_Transformer[i][1](out + x)
        x = self.attention_patch_embed[i](x)
        
        return x

    def forward_attention_CNN(self, i, x, mask):
        #pdb.set_trace()
        x_size = (x.shape[2], x.shape[3])
        mask = F.interpolate(mask, size=x_size, mode='bilinear', align_corners=False)
        out = torch.mul(x, mask)
        out = self.attention_CNN[i][0](out)
        x = self.attention_CNN[i][1](out + x)
        
        return x

    #Bilateral Fusion
    def forward_fusion(self, i, x, x_transformer):
        #pdb.set_trace()
        x_size = (x.shape[2], x.shape[3])
        x_dimension = x.shape[1]
        x_2 = self.fusion_patch_unembed[i](x_transformer, x_size)
        x_transformer_2 = self.fusion_patch_embed[i](x)
        #h, w = x.shape[-2:]
        #x_transformer_new = F.unfold(x_2, self.token_size, stride=self.token_size)
        #x_transformer_new = rearrange(x_transformer_new, 'b d t -> b t d')
        #x_transformer_new_2 = F.unfold(x, self.token_size, stride=self.token_size)
        #x_transformer_new_2 = rearrange(x_transformer_new_2, 'b d t -> b t d')
        
        x_f = torch.cat((x, x_2), 1)
        x_f = x_f + self.channel_fusion_block[i](x_f)
        
        x_cf, x_ct = torch.split(x_f, x_dimension, 1)
        
        
        #x_transformer_new = self.spatial_fusion_block[i][0](x_transformer_new, x_transformer_new_2) + x_transformer_new
        #x_transformer_new_2 = self.spatial_fusion_block[i][1](x_transformer_new_2, x_transformer_new) + x_transformer_new_2
        
        #x_transformer_new = self.spatial_fusion_block[i][2](x_transformer_new)
        #x_transformer_new_2 = self.spatial_fusion_block[i][3](x_transformer_new_2)
        
        x_transformer = self.spatial_fusion_block[i][0](x_transformer, x_transformer_2) + x_transformer
        x_transformer_2 = self.spatial_fusion_block[i][1](x_transformer_2, x_transformer) + x_transformer_2
        
        x_transformer = self.spatial_fusion_block[i][2](x_transformer) + x_transformer
        x_transformer_2 = self.spatial_fusion_block[i][3](x_transformer_2) + x_transformer_2
        
        
        x_st = self.fusion_patch_unembed[i](x_transformer, x_size)
        x_sf = self.fusion_patch_unembed[i](x_transformer_2, x_size)
        #x_transformer_new = rearrange(x_transformer_new, 'b t d -> b d t')
        #x_st = F.fold(x_transformer_new, (h, w), self.token_size, stride=self.token_size)
        #x_transformer_new_2 = rearrange(x_transformer_new_2, 'b t d -> b d t')
        #x_sf = F.fold(x_transformer_new_2, (h, w), self.token_size, stride=self.token_size)
        
        x_mf = torch.cat((x_sf, x_cf), 1)
        x_mt = torch.cat((x_st, x_ct), 1)
        
        x_mf = self.fusion_CNN[i](x_mf)
        x_mt = self.fusion_Transformer[i](x_mt)
        
        x_mtransformer = self.fusion_patch_embed[i](x_mt)
        
        return x_mf, x_mtransformer
  
    #Encoder and Bottleneck
    def forward_features(self, x):
        #pdb.set_trace()
        x_transformer = self.patch_embed(x)
        x_transformer = self.linear_encoding(x_transformer) 
        
        x_transformer_downsample = []
        x_downsample = []

        for i_layer in range(self.num_layers):
            x_size = (x.shape[2], x.shape[3])
            x_transformer = self.swin_layers[i_layer](x_transformer, x_size)
            x = self.CNN_layers[i_layer](x)
            x_transformer_res, x_res = x_transformer, x
            ###### bilateral fusion ######
            x, x_transformer = self.forward_fusion(i_layer, x, x_transformer)
            x = x + x_res
            x_transformer = x_transformer + x_transformer_res
            ########## downample #########
            x_transformer_downsample.append(x_transformer)
            x_downsample.append(x)
            x_transformer = self.swin_downsample_layers[i_layer](x_transformer)
            x = self.downsample_layers[i_layer](x)
        
        x_transformer = self.norm(x_transformer)  # B L C
        
        #pdb.set_trace()
        x_size = (x_downsample[-1].shape[2], x_downsample[-1].shape[3])
        pred = torch.cat((x_downsample[-1], self.pred_patch_unembed(x_transformer_downsample[-1], x_size)), 1)
        pred = self.conv_head(pred)
        pred = self.pred_head(pred)
        
        return x, x_transformer, x_downsample, x_transformer_downsample, pred

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_transformer, x_downsample, x_transformer_downsample, pred):
        mask = self.attentions(pred)
        for i_layer in range(self.num_layers_decoder):
            #pdb.set_trace()
            x_size = (x.shape[2], x.shape[3])
            x_transformer = self.swin_layers_decoder[i_layer](x_transformer, x_size)
            x = self.CNN_layers_decoder[i_layer](x)
            x_transformer_res, x_res = x_transformer, x
            ###### bilateral fusion ######
            x, x_transformer = self.forward_fusion(i_layer+self.num_layers, x, x_transformer)
            x = x + x_res
            x_transformer = x_transformer + x_transformer_res
            ########## upample ###########
            if i_layer < self.num_layers_decoder - 1:
                x_transformer = self.swin_upsample_layers[i_layer](x_transformer)
                x = self.upsample_layers[i_layer](x)
                
                x_size = (x.shape[2], x.shape[3])
                x_skip_transformer = self.forward_attention_transformer(i_layer, x_transformer_downsample[self.num_layers-1-i_layer], mask, x_size)
                x_transformer = torch.cat([x_transformer, x_skip_transformer],-1)
                x_transformer = self.swin_concat_back_dim[i_layer](x_transformer)
                
                x_skip = self.forward_attention_CNN(i_layer, x_downsample[self.num_layers-1-i_layer], mask)
                x = torch.cat([x, x_skip],1)
                x = self.concat_back_dim[i_layer](x)

        x_transformer = self.norm_up(x_transformer)  # B L C
        
        x_size = (x.shape[2], x.shape[3])
        x = torch.cat((x, self.patch_unembed(x_transformer, x_size)), 1)
        
        return self.conv_last(x)
        
    def forward(self, x):
        #self.mean = self.mean.type_as(x)
        #x = (x - self.mean) * self.img_range
        #pdb.set_trace()
        #pdb.set_trace()
        H, W = x.shape[2:]
        
        #x = self.sub_mean(x)
        x = self.head(x)
        
        identity = x

        x, x_transformer, x_downsample, x_transformer_downsample, pred = self.forward_features(x)
        x = self.forward_up_features(x, x_transformer, x_downsample, x_transformer_downsample, pred)
        
        x += identity
        x = self.tail(x)
        
        x_size = (x.shape[2], x.shape[3])
        pred = F.interpolate(pred, size=x_size, mode='bilinear', align_corners=False)
        
        #x = x / self.img_range + self.mean
        
        return x, pred