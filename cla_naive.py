import os
import torch
import torch.nn as nn
from tqdm import tqdm
from setup.utils import loadcla, loadcomadv, savefile

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def testClassifier(test_loader, model, verbose=False):
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        out = model(x)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    if verbose:
        print("The prediction accuracy on testset is {}".format(acc))
    return acc


def train_naive(train_loader, test_loader, model, args):
    
    model = model.train().to(device)

    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.num_epoch):
        # trainning
        model.train()
        ave_loss = 0
        step = 0
        for x, target in tqdm(train_loader):            
            x, target = x.to(device), target.to(device)
            loss = criterion(model(x),target)

            ave_loss = ave_loss * 0.9 + loss.item() * 0.1    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
        acc = testClassifier(test_loader, model)
        print("Epoch: [%d/%d], Average Loss: %.4f, test.Acc: %.4f" %
              (epoch + 1, args.num_epoch, ave_loss, acc))
    savefile(args.file_name, model, args.dataset)
    return model


def main(args):
    print('==> Setting root dir..')
    os.chdir(args.root)
    print('==> Loading data..')
    train_loader, test_loader = loadcomadv(args)
          
    print('==> Loading model..')
    model = loadcla(args)
    model = model.to(device)
    
    print('==> Training starts..')            
    train_naive(train_loader, test_loader, model, args)
    testClassifier(test_loader, model, verbose=True)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="cifar10")  
    parser.add_argument("-n", "--num_epoch", type=int, default=30)
    parser.add_argument("-f", "--file_name", default="/adv_cla")
    parser.add_argument("-l", "--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--root", default="/proj/STOR/yaoli/adv_jive")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--init", default=None)
    args = parser.parse_args()
    print(args)
    main(args)