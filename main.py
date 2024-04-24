import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as data
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
from utils_HSI import sample_gt, metrics, seed_worker
from datasets import get_dataset, HyperX
import os
import time
import numpy as np
import pandas as pd
import argparse
from network import generator
from network import encoder
from utils_ACE import *
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch S2ECNet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./datasets/Pavia/')

parser.add_argument('--source_name', type=str, default='paviaU',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=2,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if ""absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-3, 
                    help="Learning rate, set by the model if not specified.")
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.5)')
group_train.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--pro_dim', type=int, default=256)
group_train.add_argument('--test_stride', type=int, default=1,
                    help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--seed', type=int, default=233,
                    help='random seed ')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
parser.add_argument('--num_epoch', type=int, default=500,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.5,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=1,
                    help='multiple of of data augmentation')
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--d_se', type=int, default=64)
parser.add_argument('--lr_scheduler', type=str, default='none')

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=True,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")
args = parser.parse_args()


def evaluate(encoder, classifier, val_loader, gpu, tgt=False):
    ps = []
    ys = []
    for i,(x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(gpu)
            p1 = encoder(x1)
            p1 = classifier(p1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys==ps)*100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max()+1)
        print(results['Confusion_matrix'],'\n','TPR:', np.round(results['TPR']*100,2),'\n', 'OA:', results['Accuracy'], "kappa:", results['Kappa'])
    return acc


def evaluate_tgt(encoder, classifier, gpu, loader, modelpath_encoder, modelpath_classifier):
    saved_weight_encoder = torch.load(modelpath_encoder)
    saved_weight_classifier = torch.load(modelpath_classifier)
    encoder.load_state_dict(saved_weight_encoder['Encoder'])
    classifier.load_state_dict(saved_weight_classifier['Classifier'])
    encoder.eval()
    classifier.eval()
    teacc = evaluate(encoder, classifier, loader, gpu, tgt=True)
    return teacc

def experiment():
    settings = locals().copy()
    print(settings)
    hyperparams = vars(args)
    print(hyperparams)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name+'to'+args.target_name)
    log_dir = os.path.join(root, str(args.lr)+'_dim'+str(args.pro_dim)+
                           '_pt'+str(args.patch_size)+'_bs'+str(args.batch_size)+'_'+time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # writer = SummaryWriter(log_dir)
    df = pd.DataFrame([args])
    df.to_csv(os.path.join(log_dir,'params.txt'))

    seed_worker(args.seed) 
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                            args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                            args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])

    tmp = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': args.gpu, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))     

    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    if tmp < 1:
        for i in range(args.re_ratio-1):
            img_src_con = np.concatenate((img_src_con,img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con,val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True,)
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                    pin_memory=True,
                                    batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    batch_size=hyperparams['batch_size'])           
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    # 增强域生成器
    G_net = generator.Generator(num_class=num_classes, n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=args.gpu).to(args.gpu)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)
    # encoder
    Encoder = encoder.Featurizer(inchannel=N_BANDS, in_dim=args.pro_dim, patch_size=hyperparams['patch_size']).to(args.gpu)
    Encoder_opt = optim.Adam(Encoder.parameters(), lr=args.lr)
    # classifier
    Classifier = encoder.Classifier(in_dim=args.pro_dim, num_classes=num_classes).to(args.gpu)
    Classifier_opt = optim.Adam(Classifier.parameters(), lr=args.lr)

    cls_criterion = nn.CrossEntropyLoss()

    best_acc = 0
    taracc, taracc_list = 0, []
    best_taracc = 0

    for epoch in range(1,args.max_epoch+1):

        t1 = time.time()    
        loss_dict = {}

        Encoder.train()
        Classifier.train()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(args.gpu), y.to(args.gpu)
            y = y - 1
            with torch.no_grad():
                x_ED_spe, x_ED_spa = G_net(x)  
            rand = torch.nn.init.uniform_(torch.empty(len(x), 1, 1, 1)).to(args.gpu) # Uniform distribution
            x_ID_spe = rand*x + (1-rand)*x_ED_spe
            x_ID_spa = rand*x + (1-rand)*x_ED_spa
            
            x_tgt_spe, x_tgt_spa = G_net(x)
            
            Data_spe = torch.cat([x, x_ED_spe, x_ID_spe], dim=0)
            Data_spa = torch.cat([x, x_ED_spa, x_ID_spa], dim=0)

            data_list_spe = [xi for xi in Data_spe]
            data_list_spa = [xi for xi in Data_spa]

            Data_spe = torch.stack(data_list_spe, 0).to(args.gpu)  # [768, 102, 13, 13]
            Data_spa = torch.stack(data_list_spa, 0).to(args.gpu)

            target = torch.cat([y, y, y], dim=0)
            target_list = [yi for yi in target]
            target = torch.tensor(target_list).to(args.gpu)  # [768]

            bs = int(Data_spe.shape[0] / 3)

            features_spe = Encoder(Data_spe)
            features_spa = Encoder(Data_spa)

            output_spe = Classifier(features_spe)
            output_spa = Classifier(features_spa)

            objective = cls_criterion(output_spe, target) + cls_criterion(output_spa, target)
            loss_dict["objective"] = objective.item()

            sigma_spe, penalty_spe = contrastive_ace(2, Classifier, features_spe, target, num_classes, bs, args.gpu)
            sigma_spa, penalty_spa = contrastive_ace(2, Classifier, features_spa, target, num_classes, bs, args.gpu)

            alpha_spe = sigma_spe / (sigma_spe+sigma_spa)
            alpha_spa = sigma_spa / (sigma_spe+sigma_spa)

            loss_dict["penalty_spe"] = penalty_spe.item()
            loss_dict["penalty_spa"] = penalty_spa.item()

            Encoder_opt.zero_grad()
            Classifier_opt.zero_grad()

            loss = objective + alpha_spe*penalty_spe + alpha_spa*penalty_spa
            loss_dict["total_loss"] = loss.item()
            loss.backward()

            Encoder_opt.step()
            Classifier_opt.step()

            ED_features_spe = Encoder(x_tgt_spe)
            ED_features_spa = Encoder(x_tgt_spa)


            ED_output_spe = Classifier(ED_features_spe)
            ED_output_spa = Classifier(ED_features_spa)

            G_opt.zero_grad()

            ED_cls_loss = cls_criterion(ED_output_spe, y) + cls_criterion(ED_output_spa, y)
            loss_dict["ED_cls_loss"] = ED_cls_loss.item()
            ED_cls_loss.backward()

            G_opt.step()

 
            if args.lr_scheduler in ['cosine']:
                scheduler.step()
           
        loss_list = [value for value in loss_dict.values()]
        objective, penalty_spe, penalty_spa, total_loss, ED_cls_loss = loss_list

        Encoder.eval()
        Classifier.eval()

        teacc = evaluate(Encoder, Classifier, val_loader, args.gpu)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({'Encoder':Encoder.state_dict()}, os.path.join(log_dir, f'best_encoder.pkl'))
            torch.save({'Classifier':Classifier.state_dict()}, os.path.join(log_dir, f'best_classifier.pkl'))
        t2 = time.time()

        print(f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2-t1:.2f}, objective {objective:.4f} penalty_spe {penalty_spe:.4f} penalty_spa {penalty_spa:.4f} total_loss {total_loss:.4f} ED_cls_loss {ED_cls_loss:.4f} /// val {len(val_loader.dataset)}, teacc {teacc:2.2f}')
        # writer.add_scalar('objective', objective, epoch)
        # writer.add_scalar('penalty_spe', penalty_spe, epoch)
        # writer.add_scalar('penalty_spa', penalty_spa, epoch)
        # writer.add_scalar('total_loss', total_loss, epoch)
        # writer.add_scalar('ED_cls_loss', ED_cls_loss, epoch)
        # writer.add_scalar('teacc', teacc, epoch)
        
        if epoch % args.log_interval == 0:
            pklpath_encoder = f'{log_dir}/best_encoder.pkl'
            pklpath_classifier = f'{log_dir}/best_classifier.pkl'
            taracc = evaluate_tgt(Encoder, Classifier, args.gpu, test_loader, pklpath_encoder, pklpath_classifier)
            if best_taracc < taracc:
                best_taracc = taracc
                torch.save({'Best_Encoder':Encoder.state_dict()}, os.path.join(log_dir, f'best_taracc_encoder.pkl'))
                torch.save({'Best_Classifier':Classifier.state_dict()}, os.path.join(log_dir, f'best_taracc_classifier.pkl'))            
            taracc_list.append(round(taracc,2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')
    # writer.close()
    
if __name__=='__main__':
    experiment()

