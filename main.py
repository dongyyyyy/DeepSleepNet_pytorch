from models.cnn.DeepSleepNet_cnn import *
from utils.dataset.Sleep_edf.edf_to_numpy import *
from utils.dataset.Sleep_edf.makeDataset_each import *
from train.single_epoch.train_deepsleepnet import *
from train.single_epoch.train_resnet import *

from train.representation_learning.single_epoch.train_resnet_representationlearning import *
from train.representation_learning.single_epoch.train_resnet_simCLR import *

from utils.function import *

from torchsummary import summary

def make_datasets():
    # check_label()
    files = check_edf_dataset(path='/home/eslab/dataset/sleep-edf-database-expanded-1.0.0/sleep-cassette/',type='edf')
    signals_edf_list = files['signals_file_list']
    annotation_list = files['annotation_file_list']

    print(len(signals_edf_list),len(annotation_list))
    # make_dataset()
    # check_wellmade()
    # remove_unnessersary_wake()
    # makeDataset_for_loader()

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

def train_deepsleepnet_singleEpoch():
    use_channel_list = [[0],[2],[0,2],[0,1,2]]
    # use_channel_list = [[0],[0,1,2]]
    aug_p_list = [0.]
    
    aug_method_list=[[]]
    # aug_method_list=[['h_flip'],['v_flip'],['h_flip','v_flip']]
    entropy_list = [0.,0.1,.2,.3,.4,.5,.6,.7,.8,.9,1,1.5,2]
    for aug_p in aug_p_list:
        for aug_method in aug_method_list:
            for entropy_hyperparam in entropy_list:
                for use_channel in use_channel_list:
                    training_deepsleepnet_dataloader(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,random_seed=2,use_channel=use_channel,
                                                    entropy_hyperparam=entropy_hyperparam,
                                                    aug_p=aug_p,aug_method=aug_method,
                                                    classification_mode='5class',gpu_num=[0,1,2,3])


def train_resnet_singleEpoch():
    use_channel_list = [[0]]
    # use_channel_list = [[0],[0,1,2]]
    aug_p_list = [0.]
    use_model_list = ['resnet18']
    aug_method_list=[[]]
    first_conv_list = [[49, 4, 24]]
    entropy_list = [0.]
    block_kernel_size_list = [7,3,5]
    layer_filters_list = [[64,64,64,128],[64,64,128,128],[64,128,128,128],[64,64,128,256],[64,64,64,256],[64,128,128,256],[64,128,256,256],[64,128,256,512],[32,64,128,256],[16,32,64,128]]
    for layer_filters in layer_filters_list:
        for first_conv in first_conv_list:
            for block_kernel_size in block_kernel_size_list:
                for use_model in use_model_list:
                    for aug_p in aug_p_list:
                        for aug_method in aug_method_list:
                            for entropy_hyperparam in entropy_list:
                                for use_channel in use_channel_list:
                                    training_resnet_dataloader(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,random_seed=2,use_channel=use_channel,
                                                                    entropy_hyperparam=entropy_hyperparam,
                                                                    aug_p=aug_p,aug_method=aug_method,use_model = use_model,
                                                                    first_conv=first_conv,maxpool=[7,3,3], layer_filters=layer_filters,block_kernel_size=block_kernel_size,block_stride_size=2,
                                                                    classification_mode='5class',gpu_num=[0,1,2,3])

def train_resnet_singleEpoch_representationLearning():
    batch_size_list = [256,512,1024,2048,4096]
    use_channel = [0,2]
    optim_list = ['SGD','LARS']
    for optim in optim_list:
        for batch_size in batch_size_list:
            learning_rate = (0.3*batch_size / 256)
            training_resnet_dataloader_representationlearning(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,use_model = 'resnet18',
                                random_seed=2,use_channel=use_channel,entropy_hyperparam=0.,classification_mode='5class',aug_p=0.,aug_method=[],learning_rate=learning_rate,batch_size=batch_size,optim=optim,
                            first_conv=[49, 4, 24],maxpool=[7,3,3], layer_filters=[64, 64, 64, 128],block_kernel_size=3,block_stride_size=2,
                            gpu_num=[0,1,2,3])

def train_resnet_singleEpoch_representationLearning_simCLR():
    batch_size_list = [256,512,1024,2048,4096]
    use_channel = [0,2]
    optim_list = ['SGD','LARS']
    for optim in optim_list:
        for batch_size in batch_size_list:
            learning_rate = (0.3*batch_size / 256)
            training_resnet_dataloader_representationlearning_simCLR(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,use_model = 'resnet18',
                                random_seed=2,use_channel=use_channel,entropy_hyperparam=0.,classification_mode='5class',aug_p=1.,aug_method=['permute','crop'],learning_rate=learning_rate,batch_size=batch_size,optim=optim,
                                first_conv=[49, 4, 24],maxpool=[7,3,3], layer_filters=[64, 64, 64, 128],block_kernel_size=3,block_stride_size=2,
                                gpu_num=[0,1,2,3])
if __name__ == '__main__':
    train_resnet_singleEpoch_representationLearning_simCLR()
    train_resnet_singleEpoch_representationLearning()
    # train_resnet_singleEpoch()
    # check_distribution_correct()

    