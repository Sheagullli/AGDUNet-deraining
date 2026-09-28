import os
from multiprocessing import freeze_support
from matplotlib import pyplot as plt
from config import Config
import torch
from torch.backends import cudnn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from data_RGB import get_training_data, get_validation_data
from AGDGUNet import AGDUNet
# from SpaceUNet import SpaceUNet
# from TotalgradientNet import TotalgradientNet
# from AGDGUNet_plus import AGDUNet_plus
import losses

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm


def saveimg(img, name):
    from torchvision import transforms
    toPIL = transforms.ToPILImage()
    pic = toPIL(img)
    pic.save(name)


def draw_loss(Loss_list, epoch):
    plt.cla()
    x1 = range(1, epoch + 1)
    y1 = Loss_list
    fig1 = plt.figure(1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig("./Train_loss.png")
    plt.draw()
    plt.pause(4)  # 间隔的秒数： 4s
    plt.close(fig1)


if __name__ == '__main__':
    freeze_support()
    ######### 一、Set Seeds 设置种子，结果和模型的保存路径###########
    start_time = time.time()
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    opt = Config('training.yml')
    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.backends.cudnn.benchmark = True
    start_epoch = 1
    mode = opt.MODEL.MODE
    session = opt.MODEL.SESSION
    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'test_results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)
    ######### 二、Model （登录模型，导入到cuda中，并统计模型的所有参数以及需要训练的参数个数）###########
    model_restoration = AGDUNet()

    model_restoration.cuda()
    total_params = sum(p.numel() for p in model_restoration.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model_restoration.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    ######### 三、optimizer以及Scheduler（学习率调整器） ###########
    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)

    # milestone = [100,200,300,400,500,600]
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone,gamma=0.2)  # learning rates
    # scheduler.step()
    ######### 四、Resume=0 ###########
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')
    ######### 五、Loss ###########
    criterion_char = nn.MSELoss().cuda()
    # criterion_char = losses.CharbonnierLoss().cuda()
    criterion_edge = losses.EdgeLoss().cuda()

    ######### 六、DataLoaders ###########
    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4,
                              drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                            pin_memory=True)
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0
    Loss_list = []
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        # loss_edge = 0
        epoch_loss = 0
        for i, data in enumerate(tqdm(train_loader), 0):
            for param in model_restoration.parameters():
                param.grad = None
            model_restoration.train()
            optimizer.zero_grad()
            target = data[0].cuda()
            input_ = data[1].cuda()

            U0, ListU, ListR = model_restoration(input_)
            if i % 100 == 0:
                saveimg(ListU[-1][0, :, :, :], 'train_sample/background/U_%d.jpg' % i)
                saveimg(ListR[-1][0, :, :, :], 'train_sample/rain/R_%d.jpg' % i)
                # saveimg(U0[0, :, :, :], 'train_sample/U0_%d.jpg' % i)
            restored = ListU
            mask = ListR
            loss = 0.0

            loss_Bs = 0
            loss_Rs = 0
            loss_R_B = 0
            for j in range(len(restored) - 1):
                loss_Bs = loss_Bs + 0.1 * criterion_char(restored[j], target)  # 2022-09-19 fix the bug
                loss_Rs = loss_Rs + 0.1 * criterion_char(mask[j], input_ - target)  # 2022-09-19 fix the bug

            lossB = criterion_char(restored[-1], target)
            lossR = 0.9 * criterion_char(mask[-1], input_ - target)# 2022-09-19 fix the bug
            lossB0 = 0.1 * criterion_char(U0, target)# 2022-09-19 fix the bug

            loss = lossB0 + loss_Bs + lossB + loss_Rs + lossR

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        Loss_list.append(epoch_loss)
        # draw_loss(Loss_list, epoch)

        #### Evaluation ####
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                if ii > 100:
                    break

                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    U0, ListU, ListR = model_restoration(input_)
                restored = ListU[::-1]
                # if ii % 10 == 0:
                #     saveimg(restored[0, :, :, :], 'checkpoints/Deraining/test_results/AGDUNet/epoch%d_%d.jpg' % (epoch, ii))

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

            print(
                "[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))
    print("Time: {:.4f}".format(time.time() - start_time))