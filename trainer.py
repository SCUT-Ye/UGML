import os
import shutil
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather,FocalLoss,USAT,true_positive_rate
from utils.criterions import dce_evidence_u_loss
from monai.data import decollate_batch
from itertools import cycle
from Discernment.Tools import weightOptim
from monai.metrics.confusion_matrix import get_confusion_matrix,compute_confusion_matrix_metric
from monai.transforms.post.array import AsDiscrete
from networks.utils import l2_regularisation,get_soft_label


def un_loss(uncertainty,un_gt,eps=1e-5):
    # un_gt = un_gt.unsqueeze(1)
    intersection = (uncertainty * un_gt).sum()
    union = uncertainty.sum() + un_gt.sum() + eps - intersection
    similarity = intersection / union
    return similarity

def focal_loss(Dirichlet,gt):
    criterion_fl = FocalLoss(14)
    loss = criterion_fl(Dirichlet, gt)
    return loss 

def digama_loss(Dirichlet,gt,args):
    gt = gt.squeeze(1)
    Dirichlet = Dirichlet.view(Dirichlet.size(0), Dirichlet.size(1), -1)  # [N, C, HW]
    Dirichlet = Dirichlet.transpose(1, 2)  # [N, HW, C]
    Dirichlet = Dirichlet.contiguous().view(-1, Dirichlet.size(2))
    S = torch.sum(Dirichlet, dim=1, keepdim=True)
    label = F.one_hot(gt, num_classes=args.out_channels)
    label = label.view(-1, args.out_channels)
    loss = torch.sum(label * (torch.digamma(S) - torch.digamma(Dirichlet)), dim=1, keepdim=True)
    loss = torch.mean(loss)
    return loss

def KL_loss(alpha, gt,c):
    gt = gt.squeeze(1)
    alpha.view(alpha.size(0), alpha.size(1), -1) # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    label = F.one_hot(gt, num_classes=c)
    label = label.view(-1, c)
    E = alpha - 1
    alp = E * (1 - label) + 1
    S_alpha = torch.sum(alp, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alp)
    # annealing_coef = min(1, global_step / annealing_step)
    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    loss_kl = torch.mean(kl)
    return loss_kl.cuda()

def KL(alpha, gt,c,current_step, lamda_step):
    annealing_coef = min(1, current_step / lamda_step)
    gt = gt.squeeze(1)
    alpha.view(alpha.size(0), alpha.size(1), -1) # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    label = F.one_hot(gt, num_classes=c)
    label = label.view(-1, c)
    E = alpha - 1
    alp = E * (1 - label) + 1
    S_alpha = torch.sum(alp, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alp)
    # annealing_coef = min(1, global_step / annealing_step)
    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    loss_kl = torch.mean(kl)
    return annealing_coef * loss_kl.cuda()

def AU_loss(alpha, gt,c,current_step, total_step):
    eps = 1e-10
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_step * current_step)
    gt = gt.squeeze(1)
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1) # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    label = F.one_hot(gt, num_classes=c)
    label = label.view(-1, c)
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = c / S
    target = gt.view(-1, 1)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))
    
    acc_uncertain = - pred_scores * torch.log(1 - uncertainty + eps)
    inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + eps)
    L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain
    L_AU = torch.mean(L_AU)
    return L_AU,annealing_AU

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def jaccard(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    union = np.sum(np.sum(np.sum(x + y)))
    if union == 0:
        return 0.0
    return intersect / union

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, tau,args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    b_mass = np.zeros((args.sw_batch_size,args.out_channels,args.roi_x,args.roi_y,args.roi_z))
    tau = torch.from_numpy(tau).view(4, 14, 1, 1, 1).expand(4, 14, args.roi_x,args.roi_y,args.roi_z)
    tau = tau.cuda(args.rank)

    # post_label = AsDiscrete(to_onehot=args.out_channels)
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        un_gt = np.where(target == 0, 1, 0)
        un_gt = torch.from_numpy(un_gt)
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        un_gt = un_gt.cuda(args.rank)

        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            # logits = model(data,'train')
            # start_time = time.time()
            
            logits = model(data)

            # end_time = time.time() - start_time
            # print(end_time)
            # exit()
            evidences = F.softplus(logits)
            # backbone_pred = F.softmax(logits,1)
            alpha = evidences+1

            if (epoch+1) % (args.val_every)==0:
                b_mass += alpha.detach().cpu().numpy()
            alpha[alpha < tau] = 1
            S = torch.sum(alpha, dim=1, keepdim=True) #Dirichlet intensity 
            
            # E = alpha - 1 
            # b = E / (S.expand(E.shape)) # belief mass
            # print(torch.max(b))
            
            u = args.out_channels / S
            loss_un = un_loss(u,un_gt)

            # loss_diga = digama_loss(alpha,target.to(torch.int64),args)
            # loss_focal = focal_loss(alpha,target)
            loss_kl = KL(alpha,target.to(torch.int64),args.out_channels,epoch,(args.max_epochs)/2)
            # loss_AU,annealing_AU = AU_loss(alpha,target.to(torch.int64),args.out_channels,epoch,args.max_epochs)
            # loss = 0.5 * loss_func(logits, target) + 0.4 * loss_un + 0.1 * loss_kl
            # loss =  loss_func(logits, target)

            # loss =  loss_func(logits, target) + 0.3 * loss_un + 0.2 * loss_kl #组2
            loss =  0.5*loss_func(logits, target) + 0.3 * loss_un + 0.2 * loss_kl #组1
            # loss =  loss_func(logits, target) + 0.1 * loss_un + 0.2 * loss_kl #组3
            # loss =  0.7* loss_func(logits, target) + 0.1 * loss_un + 0.2 * loss_kl #组4
            # loss =   loss_func(logits, target) + 0.2 * loss_un + 0.3 * loss_kl #组5
            # loss =  0.5 *loss_func(logits, target) + 0.2 * loss_un + 0.3 * loss_kl #组6

            # loss = loss_func(logits, target) + loss_un + loss_focal
            # loss = loss_func(logits, target) + loss_diga + loss_kl*0.2 #TB
            # loss = (1 - annealing_AU)*loss_func(logits, target) + loss_diga + loss_kl + loss_AU #TMSU
            # loss = torch.mean(loss)
            # loss = dce_evidence_u_loss(target.to(torch.int64), alpha, args.out_channels, epoch, (args.max_epochs)/2,
            #                            (args.max_epochs),1e-10,False,evidences,backbone_pred)
            # loss = torch.mean(loss)
            # for PU 
            # onehot_target = get_soft_label(target, args.out_channels).permute(0, 4, 1, 2,3)
            
            # model.forward(data, onehot_target, training=True)
            # elbo = model.elbo(onehot_target)
            # reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(
            #     model.fcomb.layers)
            # loss = -elbo + 1e-5 * reg_loss
            
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "uncertainty:{:.3f}".format(torch.min(u)),
                # "alpha_avg:{:.3f}".format(torch.mean(alpha)),
                # "alpha_max:{:.3f}".format(torch.max(alpha)),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    b_mass = b_mass/(idx+1)
    return run_loss.avg,b_mass
    # return run_loss.avg

def val_epoch(model, loader, epoch, acc_func, args, model_inferer, post_label=None, post_pred=None):
    model.eval()
    start_time = time.time()
    acc = 0
    all_trp = np.zeros((args.sw_batch_size,args.out_channels))
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    post_label = AsDiscrete(to_onehot=args.out_channels)
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                    # logits = model_inferer(data,testing=True) #for PU
                    # logits = model.sample_m(data,m=8,testing=True) #for PU
                else:
                    logits = model(data,'val')
                    # logits = model.sample_m(data,m=8,testing=True) #for PU
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            pred = post_pred(logits.squeeze(0))
            pred = pred.unsqueeze(0)
            gt = post_label(target.squeeze(1))
            gt = gt.unsqueeze(0)
            matrix_1 = get_confusion_matrix(pred,gt)
            trp = compute_confusion_matrix_metric('sensitivity',matrix_1)
            all_trp += trp.detach().cpu().numpy()

            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list = distributed_all_gather([acc], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                # acc +=avg_acc

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    all_trp = np.nan_to_num(all_trp/(idx+1), nan=0.01)
    return avg_acc,all_trp
    # return avg_acc


def save_checkpoint(model, epoch, args, filename="UGML_w1_flare.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = True
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    # 判定是否启用混合精度训练
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    g_tau = np.ones((args.sw_batch_size,args.out_channels))

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        # if epoch > 30:
        #     xiWeight = 0.5*(1 + math.cos(0.1 * math.pi * (epoch - 30)))
        epoch_time = time.time()
        #  
        train_loss,b_mass = train_epoch(
                model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, tau=g_tau,args=args
            )
        if (epoch+1) % (args.val_every)==0:
            g_tau = np.mean(b_mass, axis=(2, 3, 4))
            print("tau_sum:",np.sum(g_tau))
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time: {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            # 
            val_avg_acc,all_trp = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            g_tau = USAT(g_tau,all_trp)
            print("tau_mean:",np.mean(g_tau))
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="UGML_w1_flare_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "UGML_w1_flare_final.pt"), os.path.join(args.logdir, "UGML_w1_flare.pt"))

        if scheduler is not None:
            scheduler.step()
        # if (epoch+1 ) % 10 ==0:
        #     exit()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max


def run_training_v2(
    model,
    ConsNet,
    train_loader,
    val_loader,
    all_train_loader,   #后来加的，正常训练时候可以删除
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    writer = True
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    # 判定是否启用混合精度训练
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    InterSize = len(train_loader)
    train_loader = cycle(train_loader)
    # Data Discernment 所需要的东西
    xi = 0.05
    xiRate = 1e-4
    xiWeight = 1.0
    tempLossInternal = torch.ones(InterSize).cuda(args.rank)
    tempLossExternal = torch.ones(len(all_train_loader)).cuda(args.rank)
    batch_size = args.batch_size
    idxInter = 0
    
    for epoch in range(start_epoch, args.max_epochs):
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        run_loss = AverageMeter()
        model.train()
        if epoch > 240:
            xiWeight = 0.5*(1 + math.cos(0.1 * math.pi * (epoch - 30)))
        if epoch < 80:
            # continue
            start_time = time.time()
            for idx, batch_data in enumerate(all_train_loader,0):
                data, target, flag = batch_data["image"], batch_data["label"], batch_data["flag"]
                data, target = data.cuda(args.rank), target.cuda(args.rank)
                sampleNum = data.size(0)
                
                
                logits = model(data)
                loss = loss_func(logits, target)
                tempLossExternal[idx * args.batch_size:idx * args.batch_size + sampleNum] = loss.detach()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                run_loss.update(loss.item(), n=args.batch_size)

                if args.rank == 0:
                    print(
                        "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(all_train_loader)),
                        "loss: {:.4f}".format(run_loss.avg),
                        "time {:.2f}s".format(time.time() - start_time),
                    )
                start_time = time.time()
            
        elif (epoch >= 80) & (epoch < 160):
            start_time = time.time()
            for idx, batch_data in enumerate(all_train_loader,0):
                data, target, flag = batch_data["image"], batch_data["label"], batch_data["flag"]
                data, target = data.cuda(args.rank), target.cuda(args.rank)
                for param in model.parameters():
                    param.grad = None
                with autocast(enabled=args.amp):
                    logits = model(data)
                    loss = loss_func(logits, target)
                    weight = torch.ones_like(loss)
                if flag.sum() > 0:
                    batch_dataVal = next(train_loader)
                    dataVal, targetVal = batch_dataVal["image"], batch_dataVal["label"]
                    dataVal, targetVal = dataVal.cuda(args.rank), targetVal.cuda(args.rank)
                    numInter = dataVal.size(0)
                    sampleNum = data.size(0)
                    with torch.no_grad():
                        outputVal = model(dataVal)
                    lossVal = loss_func(outputVal, targetVal)
                    tempValMean = tempLossInternal.mean() - (tempLossInternal[idxInter * batch_size:idxInter * batch_size + numInter] - lossVal).mean()
                    tempExter = tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] - loss.detach()
                    cost = abs((loss.detach() - tempValMean) * tempExter)
                    costReal = cost[flag > 0]
                    tempLossInternal[idxInter * batch_size: idxInter * batch_size + numInter] = lossVal
                    tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] = loss.detach()
                    if idxInter * batch_size + numInter > InterSize - 1:
                        idxInter = 0
                    else:
                        idxInter += 1
                    weightTemp = weightOptim(cost=costReal.numpy(), balance=args.reg_weight)
                    weight[flag > 0] = torch.from_numpy(weightTemp).type(torch.float)
                loss = (loss * weight).mean()
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                run_loss.update(loss.item(), n=args.batch_size)

                if args.rank == 0:
                    print(
                        "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(all_train_loader)),
                        "loss: {:.4f}".format(run_loss.avg),
                        "time {:.2f}s".format(time.time() - start_time),
                    )
                start_time = time.time()
        else:
            start_time = time.time()
            for idx, batch_data in enumerate(all_train_loader,0):
                data, target, flag = batch_data["image"], batch_data["label"], batch_data["flag"]
                data, target = data.cuda(args.rank), target.cuda(args.rank)
                for param in model.parameters():
                    param.grad = None
                with autocast(enabled=args.amp):
                    logits = model(data)
                    loss = loss_func(logits, target)
                    weight = torch.ones_like(loss)
                if flag.sum() > 0:
                    batch_dataVal = next(train_loader)
                    dataVal, targetVal = batch_dataVal["image"], batch_dataVal["label"]
                    dataVal, targetVal = dataVal.cuda(args.rank), targetVal.cuda(args.rank)
                    numInter = dataVal.size(0)
                    sampleNum = data.size(0)
                    # 验证模式
                    with torch.no_grad():
                        outputsVal = model(dataVal)
                    lossVal = loss_func(outputsVal, targetVal)
                    tempValMean = tempLossInternal.mean() - (tempLossInternal[idxInter * batch_size:idxInter * batch_size + numInter] - lossVal).mean()
                    tempExter = tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] - loss.detach()
                    cost = abs((loss.detach() - tempValMean) * tempExter)
                    costReal = cost[flag > 0]
                    tempLossInternal[idxInter * batch_size: idxInter * batch_size + numInter] = lossVal
                    tempLossExternal[idx * batch_size: idx * batch_size + sampleNum] = loss.detach()
                    if idxInter * batch_size + numInter > InterSize - 1:
                        idxInter = 0
                    else:
                        idxInter += 1
                    weightTemp = weightOptim(cost=costReal.numpy(), balance=args.reg_weight)
                    weight[flag > 0] = torch.from_numpy(weightTemp).type(torch.float)
                
                if flag.sum() < data.size(0):
                    with torch.no_grad():
                        outputCons = ConsNet(data)
                    lossCon = loss_func(outputCons,target)
                    lossCon = (loss - lossCon) * (1 - flag)
                    lossCon = lossCon[lossCon > 0]
                    if len(lossCon) > 0:
                        lossCon = lossCon.mean()
                        xiUpdate = lossCon.detach().item()
                    else:
                        lossCon = 0
                        xiUpdate = 0
                else:
                    lossCon = 0
                    xiUpdate = 0
                loss = (weight * loss).mean() + xiWeight * xi * lossCon
                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                xi += xiRate * xiUpdate
                run_loss.update(loss.item(), n=args.batch_size)
                if args.rank == 0:
                        print(
                            "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(all_train_loader)),
                            "loss: {:.4f}".format(run_loss.avg),
                            "time {:.2f}s".format((time.time() - start_time)),
                        )
                start_time = time.time()
            # 梯度清零
            for param in model.parameters():
                param.grad = None
        print("Epoch: %02d  ||  Loss: %.4f  ||  Time elapsed: %.2f(min)"
              % (epoch + 1, run_loss.sum, (time.time() - epoch_time) / 60))
    save_checkpoint(model, epoch, args, best_acc=1.0, filename="nnUNet_data_discernment.pt")
    print("Training Finished !")

    return val_acc_max