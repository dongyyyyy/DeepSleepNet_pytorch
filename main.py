from models.cnn.DeepSleepNet_cnn import *
from utils.dataset.Sleep_edf.edf_to_numpy import *
from utils.dataset.Sleep_edf.makeDataset_each import *
from train.single_epoch.train_deepsleepnet import *
from utils.function import *

from torchsummary import summary

def make_datasets():
    check_label()
    make_dataset()
    check_wellmade()
    remove_unnessersary_wake()
    makeDataset_for_loader()

def check_distribution_correct():
    data_path = '/home/eslab/dataset/sleep_edf/origin_npy/remove_wake_version0/each/'
    patient_list = os.listdir(data_path)
    print(len(patient_list))

    patient_list = [data_path + filename for filename in patient_list]

    labels, labels_percent = check_label_info_withPath(patient_list)
    print(labels)
    print(labels_percent)

    original_path =  '/home/eslab/dataset/sleep_edf/annotations/remove_wake_version0/'

    annotation_list = search_signals_npy(original_path)
    # print(len(annotation_list))
    labels = np.zeros(6)
    for filename in annotation_list:
        label = np.load(original_path + filename)
        labels += np.bincount(label,minlength=6)
    print(labels)
    print(labels/np.sum(labels))

if __name__ == '__main__':
    use_channel_list = [[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]]
    for use_channel in use_channel_list:
        training_deepsleepnet_dataloader(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,random_seed=2,use_channel=use_channel,entropy_hyperparam=0.,classification_mode='6class',gpu_num=[0,1,2,3])
    # check_distribution_correct()

    