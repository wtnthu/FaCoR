import gc, pdb
from sklearn.metrics import roc_curve,auc
from dataset import *
from models import *
from utils import *
import torch_resnet101
import torch
from torch.optim import SGD
from losses import *
import sys
import argparse
import scipy.io as sio 



def l2_norm(input,dim=1):
    # pdb.set_trace()
    norm = torch.norm(input,2,dim,True)
    output = torch.div(input, norm)
    return output


def find(args):
    path=args.save_path
    batch_size=args.batch_size
    log_path=args.log_path
    arch=args.arch
    aug = args.aug
    txt = args.txt
    # pdb.set_trace()
    method=args.method

    val_dataset = FIW(os.path.join(args.sample,"val_A.txt"))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    if arch=='ada3':
        model=Net_ada3().cuda()
   
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        auc ,threshold = val_model(model, val_loader, aug)
    mylog("auc : ",auc,path=log_path)
    mylog("threshold :" ,threshold,path=log_path)


def val_model(model, val_loader, aug):
    y_true = []
    y_pred = []
    # for img1, img2, img1_1, img2_1, _ in val_loader:
    method=args.method
    for data in enumerate(val_loader):
        img1, img2, labels, _ = data[1]
        # img1, img2, labels, _, _ = data[1]   
        e1,e2,x1,x2,_=model([img1.cuda(),img2.cuda()])  
        aug=False
        if aug and args.method=='mixco':
            # e1,e2,x1,x2,att= model([img1.cuda(),img2.cuda()], aug=False)
            # lambda_ = np.random.beta(0.5, 0.5, x1.shape[0])
            # lambda_ = torch.from_numpy(lambda_).type(torch.float).cuda().unsqueeze(1)
            lambda_=args.lam2
            e1_a = e1*lambda_ + (1-lambda_)*e2
            e2_a = e1*(1-lambda_) + lambda_*e2
            e1 = l2_norm(e1, dim=1)
            e2 = l2_norm(e2, dim=1)
            e1_a = l2_norm(e1_a, dim=1)
            e2_a = l2_norm(e2_a, dim=1)
            e1 = torch.cat([e1, e1_a], 1)
            e2 = torch.cat([e2, e2_a], 1) 
            y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())  
        elif aug:
            lam = np.random.beta(1, 1)
            # lam2 = np.random.beta(0.1, 0.1, img1.size()[0])
            # lam2 = torch.from_numpy(lam2).type(torch.float).cuda().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)
            # pdb.set_trace()
            lam2=args.lam2
            img1[:, :, bbx1:bbx2, bby1:bby2] = lam2*img1[:, :, bbx1:bbx2, bby1:bby2] + (1-lam2)*img2[:, :, bbx1:bbx2, bby1:bby2]
            img2[:, :, bbx1:bbx2, bby1:bby2] = lam2*img2[:, :, bbx1:bbx2, bby1:bby2] + (1-lam2)*img1[:, :, bbx1:bbx2, bby1:bby2]
            e1_2,e2_2,x1,x2,_=model([img1.cuda(),img2.cuda()])
            e1 = torch.cat([e1, e1_2], 1)
            e2 = torch.cat([e2, e2_2], 1)
            

        # e1,e2,x1,x2,_=model([img1.cuda(),img2.cuda(),img1_1.cuda(),img2_1.cuda()])    
        # y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        # if args.method=='cont':
        #     y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        if args.method=='sig':
            y_pred.extend(x1.cpu().detach().numpy().tolist())
        else:
            y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        # pdb.set_trace()
        if args.method=='rec':
            # pdb.set_trace()
            y_pred.extend(torch.max(x1, 1)[0].cpu().detach().numpy().tolist())
        # y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())
        y_true.extend(labels.cpu().detach().numpy().tolist())
    # pdb.set_trace()
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    #pdb.set_trace()
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds_keras[maxindex]
    sio.savemat(args.log_path[:-4]+'.mat', {'threshold': threshold, 'fpr': fpr, 'tpr': tpr, 'thresholds_keras': thresholds_keras})
    return auc(fpr,tpr),threshold


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="find threshold")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--log_path", type=str, default="./log.txt",help="log path default log.txt ")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    parser.add_argument( "--training_method", default="sig", type=str, help="gpu id you use")
    parser.add_argument( "--arch", default="org", type=str, help="gpu id you use")
    parser.add_argument( "--method", default="cont", type=str, help="gpu id you use")
    parser.add_argument( "--aug", default="False", type=str, help="gpu id you use")
    parser.add_argument( "--lam2", default=0.8, type=float, help="beta default 0.08")
    parser.add_argument( "--txt",  type=str, help="model save path")
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_start_method("spawn")
    set_seed(100)
    find(args)
