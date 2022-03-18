import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger

import models.anynet
import random
from skimage.metrics import structural_similarity as scikit_ssim

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='dataset/', help='datapath')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6, help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=4,help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet', help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
args = parser.parse_args(args=[])

def load_model(save_path="results/pretrained_anynet", with_spn=False, spn_init_channels=8):
    global args
    args.save_path = save_path
    args.with_spn = with_spn
    args.spn_init_channels = spn_init_channels
    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if False: # Varo - doesnt work for SPN models
        checkpoint = torch.load(args.save_path + '/checkpoint.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
    else: # works with both - with and without SPN
        print ("Model's state_dict ...")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size() )
        checkpoint = torch.load(args.save_path + '/checkpoint.tar')
        model.load_state_dict(checkpoint['state_dict'])
    return model

def ssim(im1, im2):
    mean1 = np.mean(im1)
    mean2 = np.mean(im2)
    var1 = np.var(im1)
    var2 = np.var(im2)
    covar = np.mean((im1-mean1)*(im2-mean2))
    c1 = np.square(0.01 * 255)
    c2 = np.square(0.03 * 255)
    num = (2 * mean1 * mean2 + c1) * (2 * covar + c2)
    denom = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
    val = num / denom
    return val

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def calc_ssim(model, dataloader):
    ssim_arr, ssim_scikit_arr = [], []
    for batch_idx, (imgL, imgR, dispL) in enumerate(dataloader):
        batch_size = len(imgL)
        for i in range(batch_size):
            li = imgL[i]
            ri = imgR[i]
            dl = dispL[i]
            outputs_cuda = model(imgL, imgR)
            outputs = outputs_cuda[0].cpu()
            pred_disp = outputs[i].detach()
            im1 = dl.numpy()
            im2 = pred_disp[0].numpy()
            min_im1, max_im1 = im1.shape[0], im1.shape[1]
            min_im2, max_im2 = im2.shape[0], im2.shape[1]
            m0 = min(min_im1, min_im2)
            m1 = min(max_im1, max_im2)
            im1 = im1[:m0, :m1]
            im2 = im2[:m0, :m1]
            ssim_arr.append( ssim(im1, im2) )
            ssim_scikit_arr.append( scikit_ssim(im1, im2) )
    return ssim_arr, ssim_scikit_arr


def calc_3pe_standalone(disp_src, disp_dst):
    """ 
    https://gist.github.com/MiaoDX/8d5f49c2ccb39d7f2cb8d4e57c3ab752
    https://github.com/JiaRenChang/PSMNet/issues/58 
    """
    assert disp_src.shape == disp_dst.shape, "{}, {}".format(
        disp_src.shape, disp_dst.shape)
    assert len(disp_src.shape) == 2  # (N*M)

    not_empty = (disp_src > 0) & (~np.isnan(disp_src)) & (disp_dst > 0) & (
        ~np.isnan(disp_dst))

    disp_src_flatten = disp_src[not_empty].flatten().astype(np.float32)
    disp_dst_flatten = disp_dst[not_empty].flatten().astype(np.float32)

    disp_diff_l = abs(disp_src_flatten - disp_dst_flatten)

    accept_3p = (disp_diff_l <= 3) | (disp_diff_l <= disp_dst_flatten * 0.05)
    err_3p = 1 - np.count_nonzero(accept_3p) / len(disp_diff_l)

    return err_3p


def calc_3pe(model, dataloader):
    arr = []
    for batch_idx, (imgL, imgR, dispL) in enumerate(dataloader):
        batch_size = len(imgL)
        for i in range(batch_size):
            li = imgL[i]
            ri = imgR[i]
            dl = dispL[i]
            outputs_cuda = model(imgL, imgR)
            outputs = outputs_cuda[0].cpu()
            pred_disp = outputs[i].detach()
            im1 = dl.numpy()
            im2 = pred_disp[0].numpy()
            min_im1, max_im1 = im1.shape[0], im1.shape[1]
            min_im2, max_im2 = im2.shape[0], im2.shape[1]
            m0 = min(min_im1, min_im2)
            m1 = min(max_im1, max_im2)
            im1 = im1[:m0, :m1]
            im2 = im2[:m0, :m1]
            arr.append( calc_3pe_standalone(im1, im2) )
    return arr

cmap = plt.cm.jet
cmap.set_bad(color="black")

def save_plots(model, dataloader, nname, batch_size):
    img_batch = next(iter(dataloader))
    imgL, imgR, disp_L = img_batch
    outputs_cuda = model(imgL, imgR)
    outputs = outputs_cuda[0].cpu()
    fig, ax = plt.subplots(batch_size, 4, figsize=(50, 50))
    for i in range(len(imgL)):
        li = imgL[i]
        ri = imgR[i]
        dl = disp_L[i]
        pred_disp = outputs[i].detach()
        ax[i, 0].imshow(li.permute(1, 2, 0))
        ax[i, 1].imshow(ri.permute(1, 2, 0))
        ax[i, 2].imshow(dl, cmap=cmap)
        ax[i, 3].imshow(pred_disp[0], cmap=cmap)
    plt.savefig("out_"+nname+".png")

def set_dataloaders(train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp):
    test_dataloader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, True),
        batch_size=args.train_bsize, shuffle=False, num_workers=4, drop_last=False)
    train_dataloader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=4, shuffle=False, num_workers=4, drop_last=False)
    return train_dataloader, test_dataloader

def eval_model(nname, save_path, with_spn, spn_init_channels, 
                train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp
                ):
    model = load_model(save_path, with_spn, spn_init_channels)
    train_dataloader, test_dataloader = set_dataloaders(train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp)
    save_plots(model, train_dataloader, nname+"_train", batch_size=6)
    save_plots(model, test_dataloader, nname"_test", batch_size=4)
    val_ssim, val_ssim_scikit = calc_ssim(model, test_dataloader)
    tpe = calc_3pe(model, test_dataloader)
    return val_ssim, val_ssim_scikit, tpe

def calc_stats(arr):
    res = []
    res.append(np.min(arr))
    res.append(np.mean(arr))
    res.append(np.max(arr))
    var = sum([(x - res[0])**2 for x in arr])/len(arr)
    res.append(var)
    std_dev = var ** 0.5
    res.append(std_dev)
    return res

def main():
    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader('dataset/')
    
    ssim_with_spn_8ch, ssim_sk_with_spn_8ch, tpe_with_spn_8ch = eval_model("spn_8ch", "results_with_spn_8ch/pretrained_anynet", True, 8, \
                train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp)

    ssim_with_spn_4ch, ssim_sk_with_spn_4ch, tpe_with_spn_4ch = eval_model("spn_4ch", "results_with_spn_4ch/pretrained_anynet", True, 4, \
                train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp)

    ssim_with_spn_2ch, ssim_sk_with_spn_2ch, tpe_with_spn_2ch = eval_model("spn_2ch", "results_with_spn_2ch/pretrained_anynet", True, 2, \
                train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp)

    ssim_with_spn_1ch, ssim_sk_with_spn_1ch, tpe_with_spn_1ch = eval_model("spn_1ch", "results_with_spn_1ch/pretrained_anynet", True, 1, \
                train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp)

    ssim_no_spn, ssim_sk_no_spn, tpe_no_spn = eval_model("no_spn", "results_no_spn/pretrained_anynet", False, 8, \
                train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp)

    print("No SPN  : ", calc_stats(tpe_no_spn))
    print("SPN 1ch : ", calc_stats(tpe_with_spn_1ch))
    print("SPN 2ch : ", calc_stats(tpe_with_spn_2ch))
    print("SPN 4ch : ", calc_stats(tpe_with_spn_4ch))
    print("SPN 8ch : ", calc_stats(tpe_with_spn_8ch))

    labels = ["No SPN", "SPN-1ch", "SPN-2ch", "SPN-4ch", "SPN-8ch"]
    data = [tpe_no_spn, tpe_with_spn_1ch, tpe_with_spn_2ch, tpe_with_spn_4ch, tpe_with_spn_8ch]

    stats = cbook.boxplot_stats(data, labels=labels, bootstrap=10000)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6,6), sharey=True)
    axs.bxp(stats)
    axs.set_title("Three Pioxel Error Box Plot")
    plt.savefig("box_plot_3pe.png")

if __name__ == "__main__":
    main()