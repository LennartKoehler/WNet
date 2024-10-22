
def train_single_image():
    CUDA = torch.cuda.is_available()

    # Create empty lists for average N_cut losses and reconstruction losses
    n_cut_losses_avg = []
    rec_losses_avg = []

    # Squeeze k
    # squeeze = args.squeeze
    squeeze = 2
    img_size = 256
    # wnet = WNet.WNet(squeeze=squeeze, in_chans=1)
    wnet = W_swintransformer.W_swintransformer(num_classes=squeeze,
            embed_dim=96,
            img_size=img_size,
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
            pretrained_window_sizes=[0, 0, 0])
    if(CUDA):
        wnet = wnet.cuda()
    # learning_rate = 0.003
    learning_rate = 0.003
    optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)

    n_cut_losses = []
    rec_losses = []
    start_time = time.time()

    data1 = H5Dataset("data_segments_reduced.h5")[0][None, :]
    data2 = H5Dataset("data_segments_reduced.h5")[1][None, :]
    data_batch = torch.cat((data1, data2), 0)


    for epoch in range(500):
        if (epoch > 0 and epoch % 1000 == 0):
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(wnet.parameters(), lr=learning_rate)




        wnet, n_cut_loss, rec_loss = train_op(wnet, optimizer, data_batch, 1, img_size)
        n_cut_losses.append(n_cut_loss.detach())
        rec_losses.append(rec_loss.detach())
        if epoch%10 == 0:
            print("Epoch = " + str(epoch))
            print("n_cut_loss", n_cut_loss.item())
            print("rec_loss", rec_loss.item())

    n_cut_losses_avg.append(torch.mean(torch.FloatTensor(n_cut_losses)))
    rec_losses_avg.append(torch.mean(torch.FloatTensor(rec_losses)))
    print("--- %s seconds ---" % (time.time() - start_time))
    torch.save(wnet.state_dict(), "w_swin_state_dict_with_rec_loss_500_epoch.pkl")
