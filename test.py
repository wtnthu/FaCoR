import argparse
import numpy as np
import gc, pdb
from models import *
import sys
import torch
from tensorflow.keras.preprocessing import image
import os
from utils import *
import scipy.io as sio
from sklearn.metrics import roc_curve,auc
import scipy


def l2_norm(input,dim=1):
    # pdb.set_trace()
    norm = torch.norm(input,2,dim,True)
    output = torch.div(input, norm)
    return output


def baseline_model(model_path):
    arch=args.arch
    if arch=='ada3':
        model=Net_ada3().cuda()
  
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_test(sample_path,res, args):
    txt = args.txt
    test_file_path = os.path.join(sample_path,"test_A.txt")


    test=[]
    f = open(test_file_path, "r+", encoding='utf-8')
    while True:
        line=f.readline().replace('\n','')
        if not line:
            break
        else:
            test.append(line.split(' '))
    f.close()
    res['avg'][0]=len(test)
    for now in test:
        res[now[3]][0]+=1
    return test



def gen(list_tuples, batch_size):
    total=len(list_tuples)
    start=0
    while True:
        if start+batch_size<total:
            end=start+batch_size
        else:
            end=total
        batch_list=list_tuples[start:end]
        datas=[]
        labels=[]
        classes=[]
        for now in batch_list:
            datas.append([now[1],now[2]])
            labels.append(int(now[4]))
            classes.append(now[3])
        X1 = np.array([read_image(x[0]) for x in datas])
        X2 = np.array([read_image(x[1]) for x in datas])
        yield X1, X2, labels,classes,batch_list
        start=end
        if start == total:
            yield None,None,None,None,None
        gc.collect()

def gen2(list_tuples, batch_size):
    total=len(list_tuples)
    start=0
    while True:
        if start+batch_size<total:
            end=start+batch_size
        else:
            end=total
        batch_list=list_tuples[start:end]
        datas=[]
        labels=[]
        classes=[]
        for now in batch_list:
            datas.append([now[1],now[2]])
            labels.append(int(now[4]))
            classes.append(now[3])
        # X1 = np.array([read_image(x[0]) for x in datas])
        # X2 = np.array([read_image(x[1]) for x in datas])
        X3 = np.array([read_image_align(x[0]) for x in datas])
        X4 = np.array([read_image_align(x[1]) for x in datas])
        yield X3, X4, labels,classes,batch_list
        start=end
        if start == total:
            yield None,None,None,None,None
        gc.collect()


def read_image(path):
    img = image.load_img(path, target_size=(112, 112))
    img = np.array(img).astype(np.float)
    return np.transpose(img, (2, 0, 1))

def read_image_align(path):
    try:
        img = align.get_aligned_face(path)
        img = np.array(img).astype(np.float)
        img = np.transpose(img, (2, 0, 1))
    except:
        img = image.load_img(path, target_size=(112, 112))
        img = np.array(img).astype(np.float)
        img = np.transpose(img, (2, 0, 1))
    # pdb.set_trace()
    return img


def test(args):
    model_path = args.save_path
    sample_path = args.sample
    batch_size = args.batch_size
    log_path = args.log_path
    threshold = args.threshold
    arch=args.arch
    method=args.method
    aug=args.aug
    model = baseline_model(model_path)
    classes = [
        'bb', 'ss', 'sibs', 'fd', 'md', 'fs', 'ms', 'gfgd', 'gmgd', 'gfgs', 'gmgs', 'avg'
    ]
    res={}
    for n in classes:
        res[n]=[0,0]
    test_samples = get_test(sample_path, res, args)
    with torch.no_grad():
        aug=False
        # for img1, img2, img1_1, img2_1, labels, classes, batch_list in gen(test_samples, batch_size):
        y_pred, y_true, y_pred2 = [], [], []
        for img1, img2, labels, classes, batch_list in gen(test_samples, batch_size):
            if img1 is not None:

                # img1, img2, labels, classes, batch_list = data[1]
                img1 = torch.from_numpy(img1).type(torch.float).cuda()
                img2 = torch.from_numpy(img2).type(torch.float).cuda()
                em1, em2, x1, x2,_ = model([img1, img2])
            
                if aug and args.method=='mixco':
                    # e1,e2,x1,x2,att= model([img1.cuda(),img2.cuda()], aug=False)
                    # lambda_ = np.random.beta(0.5, 0.5, x1.shape[0])
                    # lambda_ = torch.from_numpy(lambda_).type(torch.float).cuda().unsqueeze(1)
                    lambda_=args.lam2
                    em1_a = em1*lambda_ + (1-lambda_)*em2
                    em2_a = em1*(1-lambda_) + lambda_*em2
                    em1 = l2_norm(em1, dim=1)
                    em2 = l2_norm(em2, dim=1)
                    em1_a = l2_norm(em1_a, dim=1)
                    em2_a = l2_norm(em2_a, dim=1)
                    em1 = torch.cat([em1, em1_a], 1)
                    em2 = torch.cat([em2, em2_a], 1) 
                    pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
                if aug:
                    lam = np.random.beta(1, 1)
                    # lam2 = np.random.beta(0.1, 0.1, img1.size()[0])
                    # lam2 = torch.from_numpy(lam2).type(torch.float).cuda().unsqueeze(1).unsqueeze(1).unsqueeze(1)
                    bbx1, bby1, bbx2, bby2 = rand_bbox(img1.size(), lam)
                    # pdb.set_trace()
                    lam2=args.lam2
                    img1[:, :, bbx1:bbx2, bby1:bby2] = lam2*img1[:, :, bbx1:bbx2, bby1:bby2] + (1-lam2)*img2[:, :, bbx1:bbx2, bby1:bby2]
                    img2[:, :, bbx1:bbx2, bby1:bby2] = lam2*img2[:, :, bbx1:bbx2, bby1:bby2] + (1-lam2)*img1[:, :, bbx1:bbx2, bby1:bby2]

                    em1_2, em2_2, x1, x2,_ = model([img1, img2])
                    em1 = torch.cat([em1, em1_2], 1)
                    em2 = torch.cat([em2, em2_2], 1)

                # em1, em2, x1, x2,_ = model([img1, img2, img1_1, img2_1])
                # pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
                if args.method=='cont':
                    # pdb.set_trace()
                    try:
                        pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
                    except:
                        pdb.set_trace()
                if args.method=='rec':
                    pred = torch.max(x1, 1)[0].cpu().detach().numpy().tolist()
                    # pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
                if args.method=='sig':
                    pred = x1.reshape(-1).cpu().detach().numpy().tolist()
                else:
                    pred = torch.cosine_similarity(em1, em2, dim=1).cpu().detach().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(labels)
                for i in range(len(pred)):
                    # pdb.set_trace()
                    if pred[i] >= threshold:
                        p = 1
                    else:
                        p = 0
                    y_pred2.extend([p])
                    if p == labels[i]:
                        res['avg'][1] += 1
                        res[classes[i]][1] += 1
            else:
                break
        fpr, tpr, thresholds_keras = roc_curve(np.array(y_true), np.array(y_pred))
    sio.savemat(args.log_path[:-4]+'_auc_'+'.mat', {'pred': y_pred2, 'true':y_true, 'AUC':auc(fpr,tpr)})

    for key in res:
        mylog(key, ':', res[key][1] / res[key][0], path=log_path)
    print('AUC='+str(auc(fpr,tpr)))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="test  accuracy")
    parser.add_argument("--sample", type=str, help="sample root")
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--threshold", type=float, default=0.01, help=" threshold ")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size default 40")
    parser.add_argument("--log_path", type=str, default="./log.txt", help="log path default log.txt ")
    parser.add_argument("--gpu", default="1", type=str, help="gpu id you use")
    parser.add_argument( "--method", default="cont", type=str, help="gpu id you use")
    parser.add_argument( "--arch", default="org", type=str, help="gpu id you use")
    parser.add_argument( "--aug", default="False", type=str, help="gpu id you use")
    parser.add_argument( "--lam2", default=0.8, type=float, help="beta default 0.08")
    parser.add_argument( "--txt",  type=str, help="model save path")
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(100)

    # pdb.set_trace()
    threshold = scipy.io.loadmat(args.log_path[:-4]+'.mat')['threshold'][0,0]
    args.threshold=threshold
    test(args)
