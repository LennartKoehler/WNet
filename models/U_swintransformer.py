from swin_transformer_v2_1d_functionality import *



class U_swintransformer(nn.Module):
    def __init__(self,
            num_classes=100,
            embed_dim=96,
            img_size=256,
            patch_size=2,
            in_chans=1,
            depths_enc=[2, 2, 2],
            num_heads_enc=[3, 6, 12],
            depths_dec=[2, 2, 2],
            num_heads_dec=[12, 6, 3],
            window_size=8, mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0, 0, 0]):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths_enc)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches, this only has to be done at the beginning, the following patches come from patchmerging
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_enc))]  # stochastic depth decay rule


        self.out_channels = self.num_classes
        self.out_resolution = img_size

        # build layers
        self.layers_encoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=patches_resolution // (2 ** i_layer),
                               depth=depths_enc[i_layer],
                               num_heads=num_heads_enc[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths_enc[:i_layer]):sum(depths_enc[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, # patch merging always except for last layer
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers_encoder.append(layer)

        self.middle_channels = self.num_features
        self.middle_resolution = patches_resolution // (2 ** (self.num_layers - 1))


        self.layers_decoder = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.num_features // 2 ** (i_layer+1)), # + 1 because it is first upsampled and then attention calculated
                               input_resolution=self.middle_resolution * (2 ** (i_layer+1)),
                               depth=depths_dec[i_layer],
                               num_heads=num_heads_dec[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths_dec[:i_layer]):sum(depths_dec[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchSeperating if (i_layer < self.num_layers) else None, # in normal swin transformer we have i_layer < self.nu_layers -1, this currently doesnt work here, why?
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers_decoder.append(layer)


        self.skip_concat = nn.ModuleList()
        for i_layer in range(self.num_layers):
            linear = nn.Linear(int(self.num_features // 2 ** (i_layer))*2, int(self.num_features // 2 ** (i_layer)))
            self.skip_concat.append(linear)


        self.norm_enc = norm_layer(self.num_features)
        self.norm_dec = norm_layer(embed_dim//2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim//2, num_classes) if num_classes > 0 else nn.Identity()


        self.apply(self._init_weights)
        for bly in self.layers_encoder:
            bly._init_respostnorm()

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
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features_encoder(self, x):
        results_encoder = []
        results_encoder.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        results_encoder.append(x)

        for layer in self.layers_encoder:
            results_encoder.append(x)
            x = layer(x)
        x = self.norm_enc(x)  # B L C
        return x, results_encoder

    def forward_features_decoder(self, x, skip_connections):
        x = self.layers_decoder[0](x) # first encoder layer

        #after the first there are skip connections
        for skip_connection, layer, skip_concat in zip(skip_connections[:-1][::-1], self.layers_decoder[1:], self.skip_concat[1:]):
            #x = torch.cat([skip_connection, x],2)
            #x = skip_concat(x) # B L 2C -> B L C
            x = layer(x)
        x = self.norm_dec(x)

        return x

    def forward(self, x):
        enc, skip_connections = self.forward_features_encoder(x)
        dec = self.forward_features_decoder(enc, skip_connections)
        x = self.head(dec)
        x = x.transpose(1,2)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers_encoder):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops * 2 # flops encoder + decoder
    





