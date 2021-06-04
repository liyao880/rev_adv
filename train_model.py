import os
import torch
import torch.nn as nn
from tqdm import tqdm
from setup.utils import loaddata, loadmodel, savefile

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainClassifier(model, train_loader, test_loader, args):
    
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
            # if (step + 1) % args.print_every == 0:
            #     print("Epoch: [%d/%d], step: [%d/%d], Average Loss: %.4f" %
            #           (epoch + 1, args.num_epoch, step + 1, len(train_loader), ave_loss))
        acc = testClassifier(test_loader, model)
        if acc > args.thrd:
            savefile(args.file_name, model, args.dataset)
            break
    savefile(args.file_name, model, args.dataset)
    return model


def testClassifier(test_loader, model):
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
    print("The prediction accuracy on testset is {}".format(acc))
    return acc


def main(args):
    print('==> Loading data..')
    train_loader, test_loader = loaddata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)

    print('==> Training starts..')            
    model = trainClassifier(model, train_loader, test_loader, args)
    # testClassifier(test_loader,model,use_cuda=use_cuda,batch_size=args.batch_size)
    

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="mnist")   
    parser.add_argument("-n", "--num_epoch", type=int, default=30)
    parser.add_argument("-f", "--file_name", default="/cnn")
    parser.add_argument("--init", default=None)
    parser.add_argument("-l", "--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--thrd", type=float, default=0.99)
    parser.add_argument("--root", default="D:/yaoli")
    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.batch_size = 100
        args.print_every = 300
    elif args.dataset == 'cifar10':
        args.batch_size = 100
        args.print_every = 250
    else:
        print('invalid dataset')
    print(args)
    main(args)
