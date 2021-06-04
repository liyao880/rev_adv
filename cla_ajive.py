import os
os.chdir('/Users/yaoli/Documents/ACADEMIC/DeepLearning/10Adv_jive/')
import random
import numpy as np
import seaborn as sns
from joblib import dump, load
import matplotlib.pyplot as plt

from jive.AJIVE import AJIVE
from sklearn.decomposition import PCA

from setup.utils_viz import savefig
from setup.constant import Paths


## Functions
def get_correct_classified(args, feat_name, n):
    if feat_name == 'natural':
        success = np.array([False]*n)
    else:
        success = np.load("./features/"+args.dataset+'/'+feat_name+'_success.npy')
    return success


def get_scree_plot(feat, n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(feat)
    
    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()    
    return


def get_V(ajive, name):
    block = ajive.blocks[name]
    V_indi = block.individual.get_UDV()[2]
    
    V_join = block.joint.get_UDV()[2]    
    return V_indi, V_join


def ajive_analysis(feat1, feat2, r1, r2, folder_name, model_name):

    init_signal_ranks = {'feat1': r1, 'feat2': r2}
    
    ### run AJIVE
    ajive = AJIVE(init_signal_ranks=init_signal_ranks,
                  n_wedin_samples=1000, n_randdir_samples=1000,
                  n_jobs=-1, store_full=False)
    
    ajive = ajive.fit({'feat1': feat1, 'feat2': feat2})
    
    rank_indi = ajive.indiv_ranks
    print("Joint rank {}, Feat1 rank {}, Feat2 rank {}".format(ajive.joint_rank,rank_indi['feat1'], rank_indi['feat2']))
    # Save Ajive model
    os.makedirs(os.path.join(Paths().results_dir, folder_name),exist_ok=True)
    dump(ajive, os.path.join(Paths().results_dir, folder_name, model_name))
    #ajive = load(os.path.join(Paths().results_dir, model_name, model_name))
    
    # diagnostic plot
    plt.figure(figsize=[10, 10])
    ajive.plot_joint_diagnostic()
    savefig(os.path.join(Paths().results_dir, folder_name, model_name+'_ajive_diagnostic.png'))
    
    # Get Projection matrices
    V1_indi, V1_join =  get_V(ajive, 'feat1')
    V2_indi, V2_join =  get_V(ajive, 'feat2')
    return ajive, V1_indi, V1_join, V2_indi, V2_join


def dist(s, feat, V_indi, method='Average'):
    score_mat = np.matmul(feat, V_indi)
    dist = np.sum((score_mat - s)**2, axis=1)
    if method == 'Average':
        dist = np.mean(np.sqrt(dist))
    elif method == 'min':
        dist = np.min(np.sqrt(dist))
    elif method == 'max':
        dist = np.max(np.sqrt(dist))
    return dist


def get_distances(x, feat1, feat2, V1_indi, V2_indi, r1_use=1, r2_use=1):
    V1_indi = V1_indi[:,0:r1_use]
    V2_indi = V2_indi[:,0:r2_use]
    s1_i = np.matmul(x, V1_indi)
    s2_i = np.matmul(x, V2_indi)
    d1 = dist(s1_i, feat1, V1_indi)
    d2 = dist(s2_i, feat2, V2_indi)
    return d1, d2


def detection_accuracy(args, feat_test, feat1_train, feat2_train, V1_indi, V2_indi, label, direction):
    correct = 0
    d1_list = []
    d2_list = []
    for i in range(feat_test.shape[0]):       
        x = feat_test[i]
        d1, d2 = get_distances(x, feat1_train, feat2_train, V1_indi, V2_indi, r1_use=args.r1, r2_use=args.r2)
        d1_list.append(d1)
        d2_list.append(d2)
        if direction == 1 and d1 < d2:
            correct += 1
        if direction == 2 and d1 > d2:
            correct += 1
    accuracy = correct/feat_test.shape[0]
    plot_sd(d1_list, d2_list, label, args.data, accuracy, args.data[direction-1])
    return correct


def plot_sd(d1_list, d2_list, label, data, accuracy, name):
    title = "Distance density for {} in Class {} ACC {:.4f}".format(name,label, accuracy)
    save_dir = os.path.join(Paths().results_dir, args.folder_name, 'Class_'+str(label)+name)
    sns.distplot(d1_list, hist=False, kde=True, color = 'blue',
                 kde_kws = {'shade': True, 'linewidth': 2}, label=args.data[0])
    sns.distplot(d2_list, hist=False, kde=True, color = 'c',
                 kde_kws = {'shade': True, 'linewidth': 2}, label=args.data[1])
    plt.legend()
    plt.title(title)
    plt.savefig(save_dir)
    plt.clf()


def main(args):
    ## Analysis Part
    ### load pre-computed data e.g. image features
    ratio = args.ratio    
    labels = np.load("./features/"+args.dataset+"/PGD_labels.npy")
    
    ### Keep only successfully attacked (remove double False)
    success1 = get_correct_classified(args, args.data[0], labels.shape[0])
    success2 = get_correct_classified(args, args.data[1], labels.shape[0])
    
    indices = success1 + success2
    feat1_ori = np.load("./features/"+args.dataset+"/feat_"+args.data[0]+".npy")[indices]
    feat2_ori = np.load("./features/"+args.dataset+"/feat_"+args.data[1]+".npy")[indices]
    labels = labels[indices]
    
    for i in range(10):   
        indices = (labels==i)
        feat1 = feat1_ori[indices]
        feat2 = feat2_ori[indices]
        
        n = feat1.shape[0]
        print("{} number of examples in class {}".format(n,i))
        avail = list(range(0,n))
        random.shuffle(avail)
        train_ind = avail[:int(n*ratio)]
        test_ind = avail[int(n*ratio):]
        
        feat1_train, feat1_test = feat1[train_ind], feat1[test_ind]
        feat2_train, feat2_test = feat2[train_ind], feat2[test_ind]
        
        ### initial signal ranks determined from PCA scree plots
        #get_scree_plot(feat1, 40)
        #get_scree_plot(feat2, 40)
        
        model_name = args.model_name + '_class' +str(i)
        ajive, V1_indi, V1_join, V2_indi, V2_join = ajive_analysis(feat1, feat2, 15, 15, args.folder_name, model_name)
        
        correct1 = detection_accuracy(args, feat1_test, feat1_train, feat2_train, V1_indi, V2_indi, i, direction=1)
        correct2 = detection_accuracy(args, feat2_test, feat1_train, feat2_train, V1_indi, V2_indi, i, direction=2)
        accuracy = (correct1 + correct2)/(feat1_test.shape[0]*2)
        print("Accuracy of detection with class {} is {:.4f}".format(i, accuracy))
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10"], default="cifar10")   
    parser.add_argument("--data1", type=str, default = "PGD", help="natural, PGD, CW, FGSM")   
    parser.add_argument("--data2", type=str, default = "CW", help="natural, PGD, CW, FGSM") 
    parser.add_argument("--ratio", type=float, default=0.8, help='Train-test split ratio')
    parser.add_argument("--r1", type=int, default=1, help='Number of individual comp used to do detection')
    parser.add_argument("--r2", type=int, default=1, help='Number of individual comp used to do detection')
    args = parser.parse_args()
    print(args)
    args.data = [args.data1, args.data2]
    args.folder_name = args.data[0]+'_'+args.data[1]
    args.model_name = args.data[0]+'_'+args.data[1]+'_'+"r1_"+str(args.r1)+"r2_"+str(args.r2)
    main(args)
    

