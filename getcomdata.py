import random
import numpy as np


def main():
    print('==> Combing data..')
    methods = ['FGSM','PGD','CW']
    data = []
    label = []
    i = 0
    for method in methods:
        adv_data = np.load("./features/"+args.dataset+'/'+method+'_adv.npy')
        data.append(adv_data)
        label.append(np.repeat(i, adv_data.shape[0]))
        i += 1
    data = np.concatenate(data)
    label = np.concatenate(label)
    
    n = label.shape[0]
    indices = list(np.arange(n))
    
    random.shuffle(indices)
    
    print("Total number of avaialable samples: {}".format(n))
    train_list = indices[:int(n*0.8)]
    test_list = indices[int(n*0.8):]
    data_train = data[train_list]
    data_test = data[test_list]
    label_train = label[train_list]
    label_test = label[test_list]
    np.save("./features/"+args.dataset+"/adv_train.npy",data_train)
    np.save("./features/"+args.dataset+"/adv_test.npy", data_test)
    np.save("./features/"+args.dataset+"/adv_train_label.npy",label_train)
    np.save("./features/"+args.dataset+"/adv_test_label.npy",label_test)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="cifar10")   
    args = parser.parse_args()
    print(args)
    main(args)
