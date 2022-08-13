from . import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:         # Conv weight init
        torch.nn.init.xavier_uniform_(m.weight.data)

def check_label_info_withPath( file_list,class_mode='6class',check='All',check_file='.npy'):
    if class_mode =='5class':
        labels = np.zeros(5,dtype=np.intc)
    elif class_mode =='6class':
        labels = np.zeros(6,dtype=np.intc)    
    else:
        labels = np.zeros(3,dtype=np.intc)
    # print(labels)
    for signals_paths in file_list:
        # print(signals_paths)
        
        signals_list = os.listdir(signals_paths)
        signals_list.sort()
        if len(signals_list) != 0:
            for signals_filename in signals_list:
                if class_mode == '5class':
                    labels[int(signals_filename.split(check_file)[0].split('_')[-1])] += 1
                elif class_mode =='6class':
                    labels[int(signals_filename.split(check_file)[0].split('_')[-1])] += 1
                else:
                    current_label = int(signals_filename.split(check_file)[0].split('_')[-1])
                    if current_label == 4:
                        labels[2] += 1
                    elif current_label == 0:
                        labels[0] += 1
                    else:
                        labels[1] += 1
    labels_sum = labels.sum()
    print(labels_sum)
    labels_percent = labels/labels_sum
    return labels, labels_percent

def check_label_info_W_NR_R(signals_path, file_list):
    labels = np.zeros(3)
    for dataset_folder in file_list:
        signals_paths = signals_path + dataset_folder+'/'
        signals_list = os.listdir(signals_paths)
        signals_list.sort()
        for signals_filename in signals_list:
            if int(signals_filename.split('.npy')[0].split('_')[-1]) == 0:
                labels[0] += 1
            elif int(signals_filename.split('.npy')[0].split('_')[-1]) == 4:
                labels[2] += 1
            else:
                labels[1] += 1
    labels_sum = labels.sum()
    print(labels_sum)
    labels_percent = labels/labels_sum
    return labels, labels_percent

def check_label_change_W_NR_R(signals_path, file_list):
    
    total_change = [0 for _ in range(6)] # W -> NR / W -> R / NR -> W / NR -> R / R -> W / R -> NR
    total_count = [0 for _ in range(6)]
    total_num = [[] for _ in range(6)]
    
    for dataset_folder in file_list:
        current_label = 0
        count = 0
        signals_paths = signals_path + dataset_folder+'/'
        signals_list = os.listdir(signals_paths)
        signals_list.sort()
        
        for index,signals_filename in enumerate(signals_list):
            if index == 0:
                if int(signals_filename.split('.npy')[0].split('_')[-1]) == 0:
                    current_label = 0
                elif int(signals_filename.split('.npy')[0].split('_')[-1]) == 4:
                    current_label = 2
                else:
                    current_label = 1
                count = 1
            else: # W -> NR / W -> R / NR -> W / NR -> R / R -> W / R -> NR
                if int(signals_filename.split('.npy')[0].split('_')[-1]) == 0: # Wake
                    if current_label == 0: # Wake 그대로 지속
                        count += 1
                    else: # NR 또는 R에서 Wake로 온 경우
                        if current_label == 1: # NR -> W
                            total_change[2] += 1 
                            total_count[2] += count
                            total_num[2].append(count)
                        else:
                            total_change[4] += 1
                            total_count[4] += count
                            total_num[4].append(count)
                        current_label = 0 # label change
                        count = 1 # count init

                elif int(signals_filename.split('.npy')[0].split('_')[-1]) == 4:
                    if current_label == 2:
                        count += 1
                    else:
                        if current_label == 0: # W -> R
                            total_change[1] += 1 
                            total_count[1] += count
                            total_num[1].append(count)
                        else: # NR -> R
                            total_change[3] += 1
                            total_count[3] += count
                            total_num[3].append(count)

                        current_label = 2 # label change
                        count = 1 # count init
                else:
                    if current_label == 1:
                        count += 1
                    else:
                        if current_label == 0: # W -> NR
                            total_change[0] += 1 
                            total_count[0] += count
                            total_num[0].append(count)
                        else: # R -> NR
                            total_change[5] += 1
                            total_count[5] += count
                            total_num[5].append(count)

                        current_label = 1 # label change
                        count = 1 # count init
    print(total_change)
    print(total_count)
    total_change = np.array(total_change)
    total_count= np.array(total_count)
    total_num = [np.array(i) for i in total_num]
    
    print(total_count/total_change)
    print(np.sum(total_count)/np.sum(total_change))

    # total_mean = total_num.mean(1)
    for index,i in enumerate(total_num):
        print(i[:100])
        print('mean : ',i.mean())
        print('std : ',i.std())
        plt.hist(i, bins=50)
        plt.savefig('/home/eslab/%d_plot.png'%index)
        plt.cla()
    
    
    # exit(1)

def interp_1d(arr,short_sample=750,long_sample=6000):
      return np.interp(
    np.arange(0,long_sample),
    np.linspace(0,long_sample,num=short_sample),
    arr)

def interp_1d_multiChannel(arr,short_sample=750,long_sample=6000):
    signals = []
    # print(arr.shape)
    if len(arr) == 1:
        return np.interp(np.arange(0,long_sample),np.linspace(0,long_sample,num=short_sample),arr.reshape(-1)).reshape(1,-1)
    for i in range(np.shape(arr)[0]):
        signals.append(np.interp(np.arange(0,long_sample),np.linspace(0,long_sample,num=short_sample),arr[i].reshape(-1)))
    
    signals = np.array(signals)

    return signals

def interp_1d_multiChannel_tensor(arr,short_sample=750,long_sample=6000):
    signals = []
    # print(arr.shape)
    if len(arr) == 1:
        return torch.nn.functional.interpolate(input=arr.reshape(-1),size=long_sample,mode='linear')
        # return np.interp(np.arange(0,long_sample),np.linspace(0,long_sample,num=short_sample),arr.reshape(-1))
    for i in range(np.shape(arr)[0]):
        signals.append(torch.nn.functional.interpolate(input=arr[i],size=long_sample,mode='linear'))
        # signals.append(np.interp(np.arange(0,long_sample),np.linspace(0,long_sample,num=short_sample),arr[i].reshape(-1)))
    
    signals = np.array(signals)

    return signals