from . import *
from .function import *
# dowload Sleep-edf dataset(public dataset)
# !pip install pyedflib (essential)
## wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/

def check_edf_dataset(path,type='edf'): # read signal and anntoation file list
    
    if type == 'edf':
        annotations_list = search_annotations_edf(path)
        signals_list = search_signals_edf(path)
    elif type == 'npy':
        # print(path.split('/')[:-1])
        annotations_path = '/'.join(path.split('/')[:-2])+'/annotations/'
        # print(f'path = {annotations_path}')
        annotations_list = search_signals_npy(annotations_path)
        signals_list = search_signals_npy(path)

    return {'signals_file_list' : signals_list, 'annotation_file_list' : annotations_list}


def make_dataset(dataset='sleepedf',path='/home/eslab/dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'): # make edf file to npy file format!
    
    files = check_edf_dataset(path=path)
    signals_edf_list = files['signals_file_list']
    annotation_list = files['annotation_file_list']

    # print(signals_edf_list)
    # print(annotation_list)
    print(f'signals file length : {len(signals_edf_list)} // annotation file length : {len(annotation_list)}')

    epoch_size = 30
    if dataset =='sleepedf':
        sample_rate = 100

    save_signals_path = '/home/eslab/dataset/sleep_edf/origin_npy/'
    save_annotations_path = '/home/eslab/dataset/sleep_edf/annotations/'

    os.makedirs(save_annotations_path,exist_ok=True)
    os.makedirs(save_signals_path,exist_ok=True)

    for filename in signals_edf_list:
        signals_filename = filename
        annotations_filename = search_correct_annotations(path,filename)[0]
        
        signals_filename = path + signals_filename
        annotations_filename = path + annotations_filename
        

        print(signals_filename,'\n',annotations_filename)
        
        _, _, annotations_header = highlevel.read_edf(annotations_filename)
        
        label = []
        for ann in annotations_header['annotations']:
            start = ann[0]
            length = ann[1]

            length = int((length) // epoch_size) # label은 30초 간격으로 사용할 것이기 때문에 30으로 나눈 값이 해당 sleep stage가 반복된 횟수이다.
            
            if ann[2] == 'Sleep stage W':
                for time in range(length):
                    label.append(0)
            elif ann[2] == 'Sleep stage 1':
                for time in range(length):
                    label.append(1)
            elif ann[2] == 'Sleep stage 2':
                for time in range(length):
                    label.append(2)
            elif ann[2] == 'Sleep stage 3':
                for time in range(length):
                    label.append(3)
            elif ann[2] == 'Sleep stage 4':
                for time in range(length):
                    label.append(3)
            elif ann[2] == 'Sleep stage R':
                for time in range(length):
                    label.append(4)
            else:
                for time in range(length):
                    label.append(5)
        label = np.array(label)
    
        signals, _, signals_header = highlevel.read_edf(signals_filename)
        
        
        signals_len = len(signals[0]) // sample_rate // epoch_size
        annotations_len = len(label)
        if signals_header['startdate'] == annotations_header['startdate']:
            print("%s file's signal & annotations start time is same"%signals_filename.split('/')[-1])
            
            if signals_len > annotations_len :
                signals = signals[:3][:annotations_len]
            elif signals_len < annotations_len :
                signals = signals[:3]
                label = label[:signals_len]
            else:
                signals = signals[:3]
            signals = np.array(signals)
            
            np.save(save_signals_path + signals_filename.split('/')[-1].split('.')[0],signals)
            np.save(save_annotations_path + annotations_filename.split('/')[-1].split('.')[0],label)
            
            if (len(signals[0])//sample_rate//epoch_size != len(label)):
                print('signals len : %d / annotations len : %d'%(len(signals[0])//sample_rate//epoch_size,len(label)))
            else:
                print('signals file and annotations file length is same!!(No problem)')
        else:
            print("%s file''s signal & annotations start time is different"%signals_filename.split('/')[-1])

def check_wellmade(path='/home/eslab/dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/',
                    path1='/home/eslab/dataset/sleep_edf/origin_npy/'): # Check if the created npy is normal or unnormal.
    files = check_edf_dataset(path=path)
    signals_edf_list = files['signals_file_list']
    annotation_edf_list = files['annotation_file_list']

    # print(signals_edf_list)
    # print(annotation_list)
    print(f'signals file length : {len(signals_edf_list)} // annotation file length : {len(annotation_edf_list)}')

    files = check_edf_dataset(path=path1,type='npy')
    signals_npy_list = files['signals_file_list']
    annotation_npy_list = files['annotation_file_list']
    print(f'signals file length : {len(signals_npy_list)} // annotation file length : {len(annotation_npy_list)}')

    for signals_filename in signals_npy_list:
        annotations_path = '/'.join(path1.split('/')[:-2])+'/annotations/'
        annotaion_filename = search_correct_npy(annotations_path,signals_filename)[0]
        if signals_filename.split('-')[:-2] != annotaion_filename.split('-')[:-2]:
            print(f'signals file({signals_filename}) and annotation file({annotation_filename}) is not matched!!!')
        

def remove_unnessersary_wake(path='/home/eslab/dataset/sleep_edf/origin_npy/'):
    files = check_edf_dataset(path=path,type='npy')
    signals_npy_list = files['signals_file_list']
    annotation_path = '/'.join(path.split('/')[:-2])+'/annotations/'
    
    save_signals_path = path + 'remove_wake/'
    save_annotations_path = annotation_path + 'remove_wake/'
    
    
    print(save_annotations_path,save_signals_path)

    os.makedirs(save_annotations_path,exist_ok=True)
    os.makedirs(save_signals_path,exist_ok=True)
    fs = 100                               
    epoch_size = 30

    check_index_size = 20

    for signal_filename in signals_npy_list:
        total_label = np.zeros([6],dtype=int)
        current_label = np.zeros([6],dtype=int)
        annotation_filename = search_correct_npy(dirname=annotation_path,filename=signal_filename)[0]
        label = np.load(annotation_path+annotation_filename)
        signal = np.load(path+signal_filename)

        current_label = np.bincount(label,minlength=6)

        if len(label) != signal.shape[1]//(fs*epoch_size):
            print(f'{signal_filename} file is fault!!!')
        
        for remove_start_index in range(0,len(label),1):
            if(np.bincount(label[remove_start_index:(remove_start_index+check_index_size)],minlength=6)[0] != check_index_size):
                break
            
        for remove_end_index in range(len(label),-1,-1,):
            #print(np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[0])
            if(np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[0] != check_index_size and np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[5] == 0 ):
                break

        # for remove_start_index in range(0,len(label),1):
        #     if(np.bincount(label[remove_start_index:(remove_start_index+check_index_size)],minlength=6)[0] + np.bincount(label[remove_start_index:(remove_start_index+check_index_size)],minlength=6)[-1] != check_index_size):
        #         if np.sum(np.bincount(label[remove_start_index:(remove_start_index+check_index_size)],minlength=6)[1:6]) > check_index_size // 2:
        #             break
            
        # for remove_end_index in range(len(label),-1,-1):
        #     #print(np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[0])
        #     if(np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[0] + np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[-1] != check_index_size ):
        #         if np.sum(np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[1:6]) > check_index_size // 2:
        #             break
        label = label[remove_start_index:remove_end_index+1]
        signal = signal[:,remove_start_index*fs*epoch_size:(remove_end_index+1)*fs*epoch_size]
        #print(np.bincount(label,minlength=6))
        # print(signal.shape)
        # print(label.shape)
        if len(label) == len(signal[0])//30//fs:
            np.save(save_signals_path+signal_filename,signal)
            np.save(save_annotations_path+annotation_filename,label)
        
        total_label = np.bincount(label,minlength=6)
        print(f'{annotation_filename} // original label distribution : {current_label} // new label distribution : {total_label}')
    

def check_label(path='/home/eslab/dataset/sleep_edf/annotations/remove_wake/',path1='/home/eslab/dataset/sleep_edf/annotations/remove_wake_version0/'):
    list1 = search_signals_npy(path)

    list2 = search_signals_npy(path1)

    # print(len(list1),len(list2))
    total_label1 = np.zeros(6)
    total_label2 = np.zeros(6)
    for index in range(len(list1)):
        label1 = np.load(path+list1[index])
        label2 = np.load(path1+list1[index])
        total_label1 += np.bincount(label1,minlength=6)
        total_label2 += np.bincount(label2,minlength=6)
        if np.bincount(label2,minlength=6)[-1] > 10: # you can decide to remove this files in your dataset. (A lot of  'Non' class...)
            print(list1[index],np.bincount(label2,minlength=6)[-1])
        '''
        === file list ===
        SC4091EC-Hypnogram.npy 11
        SC4761EP-Hypnogram.npy 20
        SC4762EG-Hypnogram.npy 148
        SC4092EC-Hypnogram.npy 12
        '''
        # print(np.bincount(label1,minlength=6), np.bincount(label2,minlength=6))
        for i in label1:
            if i not in label2:
                print('='*30)
                print(label1)
                print(label2)
                print('='*30)
                break
            
    print(total_label1/np.sum(total_label1))
    print(total_label2/np.sum(total_label2))
