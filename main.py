import argparse
import os
from functools import partial
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.nnUnet import BasicUNet
from networks.unetr import UNETR
from networks.unet1 import UNet
# from networks.vnet import VNet
from thop import profile
from networks.utils import get_soft_label
from networks.TrustworthySeg import TMSU
from networks.VNet3D import VNet
from networks.UNet3DZoo import AttUnet,Unetdrop,unAttUnet,Unet
from networks.attentionunet import AttentionUnet
from networks.probabilistic_unet import ProbabilisticUnet
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training,run_training_v2
from utils.data_utils import get_loader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description="UNet segmentation pipeline")
parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
# 数据
# parser.add_argument("--Internal_data_dir", default="D:/URE/MONAI-dev/dataset/Task01_abdomen", type=str, help="dataset directory")
parser.add_argument("--External_data_dir", default="D:/URE/MONAI-dev/dataset/flare22", type=str, help="dataset directory")
parser.add_argument("--Internal_data_dir", default="D:/URE/MONAI-dev/dataset/flare22", type=str, help="dataset directory")
# D:/URE/MONAI-dev/dataset/flare22
parser.add_argument("--Internal_json_list", default="flare.json", type=str, help="dataset json file")
parser.add_argument("--External_json_list", default="flare.json", type=str, help="dataset json file")

parser.add_argument("--Data_discernment", default=False, type=bool, help="Data discernment or not")

parser.add_argument("--pretrained_model_name", default=None, type=str, help="pretrained model name")
parser.add_argument("--save_checkpoint", default=True, help="save checkpoint during training") #11

parser.add_argument("--max_epochs", default=1000, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")

# parser.add_argument("--optim_lr", default=0.0001, type=float, help="optimization learning rate") #for PU
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
# 正则化权重
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")

parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
# AMP（Automatic Mixed Precision）是一种训练加速技术，
# 用于深度学习模型的训练过程中。
# 它通过使用不同的数值精度来加速模型训练，同时保持模型的准确性
parser.add_argument("--noamp",default=False, help="do NOT use amp for training")
parser.add_argument("--val_every", default=5, type=int, help="validation frequency")
# 分布式训练,一般为False，毕竟单卡
parser.add_argument("--distributed", default=False, help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--model_name", default="nnUNet", type=str, help="model name")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial_dims")
parser.add_argument("--features", default=(32, 64, 128, 256, 512, 32), type=int, help="features")
# parser.add_argument("--features", default=(32, 32, 64, 128, 256, 32), type=int, help="features")

# parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
# parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
# parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
# parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
# parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
# parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")

# parser.add_argument("--res_block", action="store_true", help="use residual blocks")
# parser.add_argument("--conv_block", action="store_true", help="use conv blocks")


parser.add_argument("--use_normal_dataset", default=False, help="use monai Dataset class")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=80, type=int, help="roi size in x direction")#11
parser.add_argument("--roi_y", default=80, type=int, help="roi size in y direction")#11
parser.add_argument("--roi_z", default=80, type=int, help="roi size in z direction")#11
parser.add_argument("--dropout", default=0.0, type=float, help="dropout")
parser.add_argument("--strides", default=(2,2,2,2,2), type=float, help="strides")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    # 判定是否进行分布式训练
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    # ==================================================================
    # Set Device
    # ==================================================================
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    # ==================================================================
    # data loader
    # ==================================================================
    loader,All_train_loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    # ==================================================================
    # load model
    # ==================================================================
    if (args.model_name is None) or args.model_name == "nnUNet":
        print(args.model_name)
        model = BasicUNet(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            features=args.features,
        )
        # model = TMSU(args)
        # model = UNETR(
        #     in_channels=args.in_channels,
        #     out_channels=args.out_channels,
        #     img_size=(args.roi_x, args.roi_y, args.roi_z),
        #     feature_size=args.feature_size,
        #     hidden_size=args.hidden_size,
        #     mlp_dim=args.mlp_dim,
        #     num_heads=args.num_heads,
        #     pos_embed=args.pos_embed,
        #     norm_name=args.norm_name,
        #     conv_block=True,
        #     res_block=True,
        #     dropout_rate=args.dropout_rate,
        # )
        # model = AttUnet(
        #     in_channels=args.in_channels,
        #     base_channels=32,
        #     num_classes=args.out_channels,
        # )

        # model= VNet(
        #     spatial_dims=args.spatial_dims,
        #     in_channels=args.in_channels,
        #     out_channels=args.out_channels,
        # )
        # print(model)
        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")
        else:
            print("Do not use pretrained weights")
        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    elif (args.model_name is None) or args.model_name == "PU":    
        print(args.model_name)
        model = ProbabilisticUnet(
            input_channels=args.in_channels,
            num_classes=args.out_channels,)
        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")
        else:
            print("Do not use pretrained weights")
        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    elif (args.model_name is None) or args.model_name == "UNet":  #UE1/UE2/UE3/UE4/UE5/UNet
        print(args.model_name)
        model = UNet(
            in_channels=args.in_channels,
            base_channels=32,
            num_classes=args.out_channels,
        )
        if args.resume_ckpt:
            model_dict = torch.load(os.path.join(pretrained_dir, args.pretrained_model_name))
            model.load_state_dict(model_dict)
            print("Use pretrained weights")
        else:
            print("Do not use pretrained weights")
        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    # constraint net
    # if args.Data_discernment:
    #     ConsNet = BasicUNet(spatial_dims=args.spatial_dims, in_channels=args.in_channels,out_channels=args.out_channels,
    #                         features=args.features,dropout=args.dropout,)
    #     ConsName = "D:\\URE\\project\\runs\\nnUnet_20.pt"
    #     ConsNet_dict = torch.load(ConsName)
    #     ConsNet.load_state_dict(ConsNet_dict['state_dict'])
    #     ConsNet.cuda(args.gpu)
    #     ConsNet.eval()

    # DiceCELoss
    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    # 离散化
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    # input_t = torch.randn(4,1,80,80,80)
    # target = torch.randn(4,1,80,80,80).cuda()
    # onehot_target = get_soft_label(target, args.out_channels).permute(0, 4, 1, 2,3)

    # flops,params = profile(model,inputs=(input_t,))
    # print("Total parameters count: %.2fM" % (params / 1e6))
    # print("Total FLOPs count: %.2fG" % (flops / 1e9))
    # exit()
    best_acc = 0
    start_epoch = 0
    # ==================================================================
    # 加载预训练的权重（如果需要）
    # ==================================================================
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)
    # ==================================================================
    # 设置分布式训练（如果需要）
    # ==================================================================
    # if args.distributed:
    #     torch.cuda.set_device(args.gpu)
    #     if args.norm_name == "batch":
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model.cuda(args.gpu)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
    #     )
    # ==================================================================
    # 设置优化器
    # ==================================================================
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))
    # ==================================================================
    # 设置学习策略
    # ==================================================================
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy

    # accuracy = run_training_v2(
    #     model=model,
    #     ConsNet=ConsNet,
    #     train_loader=loader[0],
    #     val_loader=loader[1],
    #     all_train_loader=All_train_loader,
    #     optimizer=optimizer,
    #     loss_func=dice_loss,
    #     acc_func=dice_acc,
    #     args=args,
    #     model_inferer=model_inferer,
    #     scheduler=scheduler,
    #     start_epoch=start_epoch,
    #     post_label=post_label,
    #     post_pred=post_pred,
    # )
    # return accuracy


if __name__ == "__main__":
    main()
