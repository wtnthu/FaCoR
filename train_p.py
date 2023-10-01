import gc, pdb
from random import shuffle
from sklearn.metrics import roc_curve,auc
from dataset_p import *
from dataset import *
from models import *
from torch.optim import SGD
from losses import *
import argparse
import torch.optim.lr_scheduler as lr_scheduler


def l2_norm(input,dim=1):
    # pdb.set_trace()
    norm = torch.norm(input,2,dim,True)
    output = torch.div(input, norm)
    return output


def training(args):


    batch_size=args.batch_size
    val_batch_size=args.batch_size
    epochs=args.epochs
    steps_per_epoch=50
    save_path=args.save_path
    beta=args.beta
    log_path=args.log_path
    method = args.method
    arch = args.arch
    aug=args.aug
    txt = args.txt

    train_dataset=FIW2(os.path.join(args.sample,args.txt))
    val_dataset=FIW2(os.path.join(args.sample,"val_choose_A_m.txt"))
    train_loader=DataLoader(train_dataset,batch_size=batch_size,num_workers=0,pin_memory=False, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=0, pin_memory=False)
    
    if arch=='ada3':
        model=Net_ada3().cuda()

    model = torch.nn.DataParallel(model).cuda()
    optimizer_model = SGD(model.parameters(), lr=args.lr, momentum=0.9)

    max_auc=0.0


    for epoch_i in range(epochs):
        mylog("\n*************",path=log_path)
        mylog('epoch ' + str(epoch_i+ 1),path=log_path)
        contrastive_loss_epoch = 0
        model.train()

        for index_i, data in enumerate(train_loader):
            image1, image2,  labels, kin_label,_ = data

            if method=='cont':
                e1,e2,x1,x2,att= model([image1,image2], aug=False)
                beta = att
                loss = contrastive_loss(x1,x2,beta=beta)


            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()      

            contrastive_loss_epoch += loss.item()
            
            if (index_i+1)==steps_per_epoch and args.all:
                break

        use_sample=(epoch_i+1)*batch_size*steps_per_epoch
        # train_dataset.set_bias(use_sample)


        mylog("contrastive_loss:" + "%.6f" % (contrastive_loss_epoch / steps_per_epoch),path=log_path)
        model.eval()
        with torch.no_grad():
            auc = val_model(model, val_loader, aug)
        mylog("auc is %.6f "% auc,path=log_path)
        if max_auc < auc:
            mylog("auc improve from :" + "%.6f" % max_auc + " to %.6f" % auc,path=log_path)
            max_auc=auc
            mylog("save model " + save_path,path=log_path)
            save_model(model,save_path)
            # pdb.set_trace()
        else:
            mylog("auc did not improve from %.6f" % float(max_auc),path=log_path)
            # save_model(model,save_path[:-4]+'_'+str(epoch_i))
             
def save_model(model,path):
    torch.save(model.state_dict(),path)

def val_model(model, val_loader, aug):
    y_true = []
    y_pred = []
    # for img1, img2, labels, _ in val_loader:
    for img1, img2, labels, _, _ in val_loader:
        e1,e2,x1,x2,_=model([img1.cuda(),img2.cuda()])

        if args.method=='sig':
            y_pred.extend(x1.cpu().detach().numpy().tolist())
        else:
            y_pred.extend(torch.cosine_similarity(e1,e2,dim=1).cpu().detach().numpy().tolist())

        y_true.extend(labels.cpu().detach().numpy().tolist())
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr,tpr)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--batch_size", type=int, default=25,help="batch size default 25")
    parser.add_argument( "--sample", type=str, help="sample root")
    parser.add_argument( "--save_path",  type=str, help="model save path")
    parser.add_argument( "--epochs", type=int,default=40, help="epochs number default 80")
    parser.add_argument( "--lr", type=float,default=1e-4, help="epochs number default 80")
    parser.add_argument( "--beta", default=0.08, type=float, help="beta default 0.08")
    parser.add_argument( "--log_path", default="./log.txt", type=str, help="log path default log.txt")
    parser.add_argument( "--gpu", default="1", type=str, help="gpu id you use")
    parser.add_argument( "--method", default="org", type=str, help="gpu id you use")
    parser.add_argument( "--arch", default="org", type=str, help="gpu id you use")
    parser.add_argument( "--txt",  type=str, help="model save path")
    parser.add_argument( "--all", type=int, default=0,help="batch size default 25")
    parser.add_argument( "--aug", default="False", type=str, help="gpu id you use")
    parser.add_argument( "--lam2", default=0.9, type=float, help="beta default 0.08")
    parser.add_argument( "--sigma", default=0.3, type=float, help="beta default 0.08")

    args = parser.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    torch.multiprocessing.set_start_method("spawn")
    set_seed(seed=100)
    training(args)
