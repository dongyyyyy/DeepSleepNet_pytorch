from . import *
from .Transform import *


def make_weights_for_balanced_classes(data_list, nclasses=6,check_file='.npy'):
    count = [0] * nclasses
    
    for data in data_list:
        count[int(data.split(check_file)[0].split('_')[-1])] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(data_list)
    for idx, val in enumerate(data_list):
        weight[idx] = weight_per_class[int(val.split(check_file)[0].split('_')[-1])]
    return weight , count


class Sleep_Dataset_withPath_sleepEDF(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                if self.classification_mode == '5class':
                    if int(signals_filename.split('.npy')[0].split('_')[-1]) != 5: # pass 'None' class
                        signals_file = signals_path+signals_filename
                        all_signals_files.append(signals_file)
                        all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))    
                else:
                    signals_file = signals_path+signals_filename
                    all_signals_files.append(signals_file)
                    all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_cuda = True,
                 use_channel = [0],
                 window_size=500,
                 stride=250,
                 sample_rate=125,
                 epoch_size=30,
                 aug_p = 0.,
                 aug_method = ['h_flip','v_flip'],
                 classification_mode='5class'
                 ):
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.classification_mode = classification_mode
        self.signals_files_path, self.labels, self.length = self.read_dataset()
        self.use_channel = use_channel
        self.use_cuda = use_cuda
        self.seq_size = ((sample_rate*epoch_size)-window_size)//stride + 1
        self.window_size = window_size
        self.stride = stride
        self.aug_p = aug_p
        self.aug_method = aug_method
        
        print('classification_mode : ',classification_mode)
        print(f'window size = {window_size} / stride = {stride}')

    def __getitem__(self, index):

        # current file index

        labels = int(self.labels[index])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1

        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2        

        signals = np.load(self.signals_files_path[index])
        signals = signals[self.use_channel,:]
        

        # for i in range(self.seq_size):
        #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

        signals = np.array(signals)
        if self.aug_p > 0.:
            if np.random.rand() > self.aug_p: #using aug
                if 'h_flip' in self.aug_method: # horizontal flip
                    signals = -1 * signals
                elif 'v_flip' in self.aug_method: # vertical flip
                    signals = signals[:,::-1]
        if self.use_cuda:
            signals = torch.from_numpy(signals).float()

        # print(signals.shape)
        return signals,labels
        
    def __len__(self):
        return self.length 



class Sleep_Dataset_withPath_sleepEDF_simCLR(object):
    def read_dataset(self):
        all_signals_files = []
        all_labels = []

        for dataset_folder in self.dataset_list:
            signals_path = dataset_folder
            signals_list = os.listdir(signals_path)
            signals_list.sort()
            for signals_filename in signals_list:
                if self.classification_mode == '5class':
                    if int(signals_filename.split('.npy')[0].split('_')[-1]) != 5: # pass 'None' class
                        signals_file = signals_path+signals_filename
                        all_signals_files.append(signals_file)
                        all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))    
                else:
                    signals_file = signals_path+signals_filename
                    all_signals_files.append(signals_file)
                    all_labels.append(int(signals_filename.split('.npy')[0].split('_')[-1]))
                
        return all_signals_files, all_labels, len(all_signals_files)

    def __init__(self,
                 dataset_list,
                 class_num=5,
                 use_cuda = True,
                 use_channel = [0],
                 window_size=500,
                 stride=250,
                 sample_rate=125,
                 epoch_size=30,
                 preprocessing = True,
                 preprocessing_method = ['permute','crop'],
                 permute_size=200,
                 crop_size=1000,
                 cutout_size=1000,
                 classification_mode='5class'
                 ):
        self.class_num = class_num
        self.dataset_list = dataset_list
        self.classification_mode = classification_mode
        self.signals_files_path, self.labels, self.length = self.read_dataset()
        self.use_channel = use_channel
        self.use_cuda = use_cuda
        self.seq_size = ((sample_rate*epoch_size)-window_size)//stride + 1
        self.window_size = window_size
        self.stride = stride
        self.long_length =sample_rate * epoch_size
        self.preprocessing = preprocessing
        self.preprocessing_method = preprocessing_method
        self.Transform = Transform()
        self.permute_size = permute_size
        self.crop_size = crop_size
        self.cutout_size = cutout_size

        print('classification_mode : ',classification_mode)
        print(f'window size = {window_size} / stride = {stride}')

    def __getitem__(self, index):

        # current file index

        labels = int(self.labels[index])

        if self.classification_mode == 'REM-NoneREM':
            if labels == 0: # Wake
                labels = 0
            elif labels == 4: #REM
                labels = 2
            else: # None-REM
                labels = 1

        elif self.classification_mode == 'LS-DS':
            if labels == 0:
                labels = 0
            elif labels == 1 or labels == 2:
                labels = 1
            else:
                labels = 2        

        signals = np.load(self.signals_files_path[index])
        signals = signals[self.use_channel,:]
        

        # for i in range(self.seq_size):
        #     print(np.array_equal(signals[:,i*self.stride:(i*self.stride)+self.window_size],input_signals[i]))              

        # signals = np.array(signals)
        if self.preprocessing:
            for signal_index, current_method in enumerate(self.preprocessing_method):
                if current_method == 'permute':
                    if signal_index == 0:
                        signals1 = self.Transform.permute(signal=signals,pieces_size=self.permute_size)
                    else:
                        signals2 = self.Transform.permute(signal=signals,pieces_size=self.permute_size)
                elif current_method == 'crop':
                    if signal_index == 0:
                        signals1 = self.Transform.crop_resize(signal=signals,length=self.crop_size,long_sample=self.long_length)
                    else:
                        signals2 = self.Transform.crop_resize(signal=signals,length=self.crop_size,long_sample=self.long_length)
                elif current_method =='cutout':
                    if signal_index == 0:
                        signals1 = self.Transform.cutout_resize(signal=signals,length=self.cutout_size)
                    else:
                        signals2 = self.Transform.cutout_resize(signal=signals,length=self.cutout_size)

        if self.use_cuda:
            signals1 = torch.from_numpy(signals1).float()
            signals2 = torch.from_numpy(signals2).float()

        # print(signals.shape)
        return signals1,signals2,labels
        
    def __len__(self):
        return self.length 

