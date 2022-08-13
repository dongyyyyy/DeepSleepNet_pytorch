from . import *
from .function import *

def makeDataset_for_loader(path='/home/eslab/dataset/sleep_edf_final/origin_npy/remove_wake_version1/'):
    train_list = search_signals_npy(dirname=path)
    train_save_path = path + 'each/'
    os.makedirs(train_save_path,exist_ok=True)

    annotation_path = path.split('/')
    annotation_path[-3] = 'annotations'

    annotation_path = '/'.join(annotation_path)
    sample_rate = 100
    epoch_size = 30
    

    for folder_name in train_list:
        annotation_filename = search_correct_npy(dirname=annotation_path,filename=folder_name)[0]

        create_folder_path = train_save_path + (folder_name.split('-')[0])[:-2] + '/'
        os.makedirs(create_folder_path,exist_ok=True)
        
        signals = np.load(path+folder_name)
        label = np.load(annotation_path + annotation_filename)

        # print(signals.shape[1]//(sample_rate*epoch_size))
        # print(label.shape)
        if signals.shape[1]//(sample_rate*epoch_size) == len(label):
            for index in range(len(label)):
                save_signals = signals[:,index*sample_rate*epoch_size:(index+1)*sample_rate*epoch_size]
                if index < 10:
                    file_name = f'000{index}'
                elif index < 100:
                    file_name = f'00{index}'
                elif index < 1000:
                    file_name = f'0{index}'
                else:
                    file_name = f'{index}'
                save_file = create_folder_path+f'{file_name}_{label[index]}'
                np.save(save_file,save_signals)
            

        # label = 

    print(len(train_list))