import os
import torch
import numpy as np
import eagerpy as ep
import foolbox as fb
from tqdm import tqdm
from setup.utils import loadmodel, randomdata
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_adv(args, fmodel, attack, test_loader):
    count = 0
    success = []
    adv_samples = []
    label_list = []
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        images, labels = ep.astensors(*(images,labels))
        _, advs, success_batch = attack(fmodel, images, labels, epsilons=[args.epsilon])
        adv_samples.append(advs[0].raw.detach().cpu().numpy())
        success.append(success_batch[0].raw.detach().cpu().numpy()[0])
        label_list.append(labels.raw.detach().cpu().numpy()[0])
        count += images.shape[0]
        if count >= args.n_samples:
            break
    adv_samples = np.concatenate(adv_samples,axis=0)
    success = np.array(success)
    label_list = np.array(label_list)
    robust_accuracy = 1.0 - success.mean(axis=-1)
    print("Method={},num.sample={}, robust.acc={:.4f}".format(args.method, args.n_samples, robust_accuracy))
    return adv_samples, success, label_list


def get_all_samples(test_loader):
    img = []
    lab = []
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        img.append(images.detach().numpy())
        lab.append(labels.detach().numpy())
    x_test = np.concatenate(img)
    lab = np.concatenate(lab)
    y_test = np.zeros((lab.size, lab.max()+1))
    y_test[np.arange(lab.size),lab] = 1
    return x_test, y_test, lab



def main(args):
    print('==> Loading data..')
    test_loader, _ = randomdata(args, np.load("./features/"+args.dataset+'/indices.npy'))
    #np.save("./features/"+args.dataset+'/indices.npy', indices)
    
    if args.n_samples == None:
        args.n_samples = len(test_loader.dataset)
    print('==> Loading model..')
    model = loadmodel(args)
    model = model.to(device)
    model = model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=args.input_shape,
        nb_classes=10,
        )
    
    if args.method == 'CW':
        attack = CarliniL2Method(classifier=classifier, max_iter=args.cw_steps)
        x_test, y_test, labels = get_all_samples(test_loader)
        adv_samples = attack.generate(x=x_test)
        predictions = classifier.predict(adv_samples)
        success = (np.argmax(predictions, axis=1) != labels)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Query:{}, and Accuracy: {:.4f}".format(args.cw_steps, accuracy))
    else:
        if args.method == 'PGD':    
            attack = fb.attacks.PGD(rel_stepsize=args.alpha, steps=args.steps)
        elif args.method == 'FGSM':
            attack = fb.attacks.FGSM()
        elif args.method == 'CW-f':
            attack = fb.attacks.L2CarliniWagnerAttack(steps=1000)
        else:
            raise ValueError
        adv_samples, success, labels = generate_adv(args, fmodel, attack, test_loader)
    np.save("./features/"+args.dataset+'/'+args.method+'_adv', adv_samples)
    np.save("./features/"+args.dataset+'/'+args.method+'_labels', labels)
    np.save("./features/"+args.dataset+'/'+args.method+'_success', success)

    # adv_samples = np.load("./features/"+args.dataset+'/'+args.method+'_adv.npy')
    # labels = np.load("./features/"+args.dataset+'/'+args.method+'_labels.npy')
    # success = np.load("./features/"+args.dataset+'/'+args.method+'_success.npy')
       
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Model Evaluation')
    parser.add_argument("-d", '--dataset', type=str, choices=["mnist", "cifar10"], default="cifar10")   
    parser.add_argument("--method", type=str, default="CW", choices=['FGSM','PGD','CW','CW-f'])
    parser.add_argument("--root", type=str, default="D:/yaoli")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--cw_steps", type=int, default=10)
    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.batch_size = 1
        args.init = "/cnn"
        args.epsilon = 0.3
        args.steps = 40
        args.alpha = 0.03
        args.input_shape = (1,28,28)
    elif args.dataset == 'cifar10':
        args.batch_size = 1
        args.init = "/vgg16"
        args.epsilon = 0.031
        args.steps = 20
        args.alpha = 0.003
        args.input_shape = (3,32,32)
    else:
        print('invalid dataset')
    print(args)
    main(args)
