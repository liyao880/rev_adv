import torch
import numpy as np
from tqdm import tqdm
from setup.utils import loadadvdata, loadmodel

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_feature(data_loader, model, args):
    model.eval()
    features = []
    for batch_idx, (x,_) in enumerate(tqdm(data_loader)):
        x = x.to(device)
        feat = model.get_feature(x)
        features.append(feat.detach().cpu().numpy())
    features = np.concatenate(features, axis=0)
    np.save("./features/"+args.dataset+'/feat_'+args.method,features)
    return


def main(args):
    print('==> Loading data..')
    data_loader = loadadvdata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)
    model = model.to(device)
    
    print('==> Training starts..')            
    generate_feature(data_loader, model, args)
    

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="cifar10")   
    parser.add_argument("--method", type=str, default="CW", choices=['natural','FGSM','PGD','CW'])
    parser.add_argument("--root", default="D:/yaoli")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.init = '/cnn'
    elif args.dataset == 'cifar10':
        args.init = '/vgg16'
    else:
        print('invalid dataset')
    print(args)
    main(args)