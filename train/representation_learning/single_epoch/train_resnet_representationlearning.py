from . import *
from models.cnn.ResNet import *



def train_resnet_dataloader_representationlearning(save_filename,logging_filename,train_dataset_list,val_dataset_list,test_dataset_list,batch_size = 512,entropy_hyperparam=0.,
                                                epochs=100,optim='Adam',loss_function='CE',use_model='resnet18',
                                                learning_rate=0.001,scheduler=None,warmup_iter=20,cosine_decay_iter=40,stop_iter=10,
                                                use_channel=[0,1],class_num=6,classification_mode='6class',aug_p=0.,aug_method=['h_flip','v_flip'],
                                                first_conv=[49, 4, 24],maxpool=[7,3,3], layer_filters=[64, 128, 128, 256],block_kernel_size=5,block_stride_size=2,
                                                gpu_num=0,sample_rate= 100,epoch_size = 30):
    # cpu processor num
    cpu_num = multiprocessing.cpu_count()
    

    #dataload Training Dataset
    train_dataset = Sleep_Dataset_withPath_sleepEDF(dataset_list=train_dataset_list,class_num=class_num,
    use_channel=use_channel,use_cuda = True,classification_mode=classification_mode,aug_p = aug_p,aug_method = aug_method,)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, num_workers=(cpu_num//4))

    # calculate weight from training dataset (for "Class Balanced Weight")
    weights,count = make_weights_for_balanced_classes(train_dataset.signals_files_path,nclasses=class_num)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights,len(weights))


    #dataload Validation Dataset
    val_dataset = Sleep_Dataset_withPath_sleepEDF(dataset_list=val_dataset_list,class_num=class_num,
    use_channel=use_channel,use_cuda = True,classification_mode=classification_mode)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=(cpu_num//4))


    test_dataset = Sleep_Dataset_withPath_sleepEDF(dataset_list=test_dataset_list,class_num=class_num,
    use_channel=use_channel,use_cuda = True,classification_mode=classification_mode)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=(cpu_num//4))


    print(train_dataset.length,val_dataset.length,test_dataset.length)
    
    # Adam optimizer paramQ
    b1 = 0.9
    b2 = 0.999

    # for Regularization
    beta = 0.001
    norm_square = 2

    check_file = open(logging_filename, 'w')  # logging file

    best_accuracy = 0.
    best_epoch = 0
    if use_model == 'resnet18':
        model = ResNet_contrastiveLearning(block=BasicBlock, layers=[2,2,2,2], first_conv=first_conv,maxpool=maxpool, layer_filters=layer_filters, in_channel=len(use_channel),
                 block_kernel_size=block_kernel_size,block_stride_size=block_stride_size, embedding=256,feature_dim=128, use_batchnorm=True, zero_init_residual=False)
    elif use_model == 'resnet34':
        model = ResNet_contrastiveLearning(block=BasicBlock, layers=[3,4,6,3], first_conv=first_conv,maxpool=maxpool, layer_filters=layer_filters, in_channel=len(use_channel),
                 block_kernel_size=block_kernel_size,block_stride_size=block_stride_size, embedding=256,feature_dim=128, use_batchnorm=True, zero_init_residual=False)
    elif use_model == 'resnet50':
        model = ResNet_contrastiveLearning(block=Bottleneck, layers=[3,4,6,3], first_conv=first_conv,maxpool=maxpool, layer_filters=layer_filters, in_channel=len(use_channel),
                 block_kernel_size=block_kernel_size,block_stride_size=block_stride_size, embedding=256,feature_dim=128, use_batchnorm=True, zero_init_residual=False)
                                   
    cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_num[0]}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    if cuda:
        print('can use CUDA!!!')
        model = model.cuda()
    summary(model,(len(use_channel),sample_rate*epoch_size))
    # exit(1)
    print('torch.cuda.device_count() : ', torch.cuda.device_count())

    if torch.cuda.device_count() > 1:
        print('Multi GPU Activation !!!', torch.cuda.device_count())
        model = nn.DataParallel(model)

    # summary(model, (3, 6000))
    model.apply(weights_init)  # weight init
    print('loss function : %s' % loss_function)

    loss_fn = SupConLoss(temperature=0.07,contrast_mode='one').to(device)

    # optimizer ADAM (SGD의 경우에는 정상적으로 학습이 진행되지 않았음)
    if optim == 'Adam':
        print('Optimizer : Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(b1, b2))
    elif optim == 'RMS':
        print('Optimizer : RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optim == 'SGD':
        print('Optimizer : SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5,nesterov=False)
    elif optim == 'AdamW':
        print('Optimizer AdamW')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(b1, b2))
    elif optim == 'LARS':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5,nesterov=False)
        optimizer = torchlars.LARS(optimizer=optimizer,eps=1e-8,trust_coef=0.001)

    gamma = 0.8

    lr = learning_rate
    epochs = epochs
    if scheduler == 'WarmUp_restart_gamma': 
        print(f'target lr : {learning_rate} / warmup_iter : {warmup_iter} / cosine_decay_iter : {cosine_decay_iter} / gamma : {gamma}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_decay_iter+1)
        scheduler = LearningRateWarmUP_restart_changeMax(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    cosine_decay_iter=cosine_decay_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine,gamma=gamma)
    elif scheduler == 'WarmUp_restart':
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_decay_iter+1)
        scheduler = LearningRateWarmUP_restart(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    cosine_decay_iter=cosine_decay_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine)
    elif scheduler == 'WarmUp':
        print(f'target lr : {learning_rate} / warmup_iter : {warmup_iter}')
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs-warmup_iter+1)
        scheduler = LearningRateWarmUP(optimizer=optimizer,
                                    warmup_iteration=warmup_iter,
                                    target_lr=lr,
                                    after_scheduler=scheduler_cosine)
    elif scheduler == 'StepLR': # 특정 epoch 도착하면 비율만큼 감소
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [11, 21], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler == 'Reduce': # factor 비율만큼 줄여주기 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.5, patience=10,
                                                           min_lr=1e-6)
    elif scheduler == 'Cosine':
        print('Cosine Scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max=epochs)


    # scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=.5)
    # loss의 값이 최소가 되도록 하며, 50번 동안 loss의 값이 감소가 되지 않을 경우 factor값 만큼
    # learning_rate의 값을 줄이고, 최저 1e-6까지 줄어들 수 있게 설정
    
    best_loss = 0.
    stop_count = 0
    check_loss = False
    for epoch in range(epochs):
        if scheduler != 'None':
            scheduler.step(epoch)
        train_total_loss = 0.0
        train_total_count = 0
        train_total_data = 0

        val_total_loss = 0.0
        val_total_count = 0
        val_total_data = 0

        start_time = time.time()
        model.train()

        output_str = 'current epoch : %d/%d / current_lr : %f \n' % (epoch+1,epochs,optimizer.state_dict()['param_groups'][0]['lr'])
        sys.stdout.write(output_str)
        check_file.write(output_str)
        with tqdm(train_dataloader,desc='Train',unit='batch') as tepoch:
            for index,(batch_signal, batch_label) in enumerate(tepoch):
                batch_signal = batch_signal.to(device)
                batch_label = batch_label.long().to(device)
                optimizer.zero_grad()

                pred = model(batch_signal)
                pred = pred.unsqueeze(1)
                loss = loss_fn(pred, batch_label)  # + beta * norm
                
                train_total_loss += loss.item()

                loss.backward()
                optimizer.step()

                
                tepoch.set_postfix(loss=train_total_loss/(index+1))
                
        train_total_loss /= index

        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f \n' \
                    % (epoch + 1, epochs, time.time() - start_time, train_total_loss)
        # sys.stdout.write(output_str)
        check_file.write(output_str)

        # check validation dataset
        start_time = time.time()
        model.eval()

        with tqdm(val_dataloader,desc='Validation',unit='batch') as tepoch:
            for index,(batch_signal, batch_label) in enumerate(tepoch):
                batch_signal = batch_signal.to(device)
                batch_label = batch_label.long().to(device)

                with torch.no_grad():
                    pred = model(batch_signal)
                    pred = pred.unsqueeze(1) # [batch , num of views(augmentation) , embedding_size]
                    loss = loss_fn(pred, batch_label)
                    # print(f'val loss : {loss.item()}')
                    val_total_loss += loss.item()
                    tepoch.set_postfix(loss=val_total_loss/(index+1))

        val_total_loss /= (index + 1)

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f\n' \
                    % (epoch + 1, epochs, time.time() - start_time, val_total_loss)
        # sys.stdout.write(output_str)
        check_file.write(output_str)
        
        # scheduler.step(float(val_total_loss))
        # scheduler.step(epoch)
        if epoch == 0:
            best_loss = val_total_loss
            best_epoch = epoch
            save_file = save_filename
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), save_file)
            else:
                torch.save(model.state_dict(), save_file)
            stop_count = 0
            # sys.stdout.write(output_str)
            check_file.write(output_str)
        else:
            if best_loss > val_total_loss:
                best_loss = val_total_loss
                best_epoch = epoch
                save_file = save_filename
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), save_file)
                else:
                    torch.save(model.state_dict(), save_file)
                stop_count = 0
            else:
                stop_count += 1
        
        output_str = 'best epoch : %d/%d / best loss : %f%%\n' \
                 % (best_epoch + 1, epochs, best_loss)
        sys.stdout.write(output_str)
        print('=' * 30)

    output_str = 'best epoch : %d/%d / best loss : %f%%\n' \
                 % (best_epoch + 1, epochs, best_loss)
    sys.stdout.write(output_str)
    check_file.write(output_str)
    print('=' * 30)

    check_file.close()





def training_resnet_dataloader_representationlearning(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,use_model = 'resnet18',
                                random_seed=2,use_channel=[0,1],entropy_hyperparam=0.,classification_mode='5class',aug_p=0.,aug_method=['h_flip','v_flip'],
                                first_conv=[49, 4, 24],maxpool=[7,3,3], layer_filters=[64, 128, 128, 256],block_kernel_size=5,block_stride_size=2,learning_rate=0.1,batch_size=512,optim='SGD',
                                gpu_num=[0]):
    
    if use_dataset == 'sleep_edf':
        signals_path = '/home/eslab/dataset/sleep_edf_final/origin_npy/remove_wake_version1/each/'

    random.seed(random_seed) # seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # signals_path = '/home/eslab/dataset/seoulDataset/1channel_prefilter_butter_minmax_-1_1/signals_dataloader/'

    dataset_list = os.listdir(signals_path)
    dataset_list = [signals_path + filename + '/'  for filename in dataset_list]
    dataset_list.sort()
    random.shuffle(dataset_list)

    

    training_fold_list = []
    validation_fold_list = []
    test_fold_list = []
    
    
    
    val_length = int(len(dataset_list) * val_percent)
    test_length = int(len(dataset_list) * test_percent)  
    train_length = int(len(dataset_list) - val_length - test_length)

        
    for i in range(0,val_length):
        validation_fold_list.append(dataset_list[i])
    for i in range(val_length,val_length + test_length):
        test_fold_list.append(dataset_list[i])
    for i in range(val_length + test_length,len(dataset_list)):
        training_fold_list.append(dataset_list[i])    
        
        
    
    # print(dataset_list[:10])
    print('='*20)
    print(len(training_fold_list))
    print(len(validation_fold_list))
    print(len(test_fold_list)) 
    print('='*20)

    train_label,train_label_percent = check_label_info_withPath(file_list = training_fold_list)
    val_label,val_label_percent = check_label_info_withPath(file_list = validation_fold_list)
    test_label,test_label_percent = check_label_info_withPath(file_list = test_fold_list)
    
    print(train_label)
    print(np.round(train_label_percent,3))
    print(val_label)
    print(np.round(val_label_percent,3))
    print(test_label)
    print(np.round(test_label_percent,3))

    
    # exit(1)
    
    # number of classes
    if classification_mode == '6class':
        class_num = 6
    elif classification_mode =='5class':
        class_num = 5
    else:   
        class_num=3

    # hyperparameters
    epochs = 100
    # batch_size = 2048
    warmup_iter=10
    cosine_decay_iter=10
    
    stop_iter = 10
    loss_function = 'CE' # CEs
    scheduler = 'WarmUp' # 'WarmUp_restart'    

    print(f'class num = {class_num}')
    model_save_path = f'/data/hdd3/git/DeepSleepNet_pytorch/saved_model/representation_learning/{use_dataset}/{classification_mode}/'\
    f'single_epoch_models_{round(train_percent,2)}_{round(val_percent,2)}_{round(test_percent,2)}/'\
    f'optim_{optim}_random_seed_{random_seed}_scheduler_{scheduler}_withoutRegularization_aug_p_{aug_p}_aug_method_{aug_method}/'\
    f'firstconv_{first_conv}_maxpool_{maxpool}_layerfilters_{layer_filters}_blockkernelsize_{block_kernel_size}_blockstridesize_{block_stride_size}/'
    logging_save_path = f'/data/hdd3/git/DeepSleepNet_pytorch/log/representation_learning/{use_dataset}/{classification_mode}/'\
    f'single_epoch_models_{round(train_percent,2)}_{round(val_percent,2)}_{round(test_percent,2)}/'\
    f'optim_{optim}_random_seed_{random_seed}_scheduler_{scheduler}_withoutRegularization_aug_p_{aug_p}_aug_method_{aug_method}/'\
    f'firstconv_{first_conv}_maxpool_{maxpool}_layerfilters_{layer_filters}_blockkernelsize_{block_kernel_size}_blockstridesize_{block_stride_size}/'
    # model_save_path = '/home/eslab/kdy/git/Sleep_pytorch/saved_model/seoulDataset/single_epoch_models/'
    # logging_save_path = '/home/eslab/kdy/git/Sleep_pytorch/log/seoulDataset/single_epoch_models/'

    os.makedirs(model_save_path,exist_ok=True)
    os.makedirs(logging_save_path,exist_ok=True)
    
    save_filename = model_save_path + f'{use_model}_%.5f_{use_channel}_{batch_size}_entropy_{entropy_hyperparam}.pth'%(learning_rate)
    
    logging_filename = logging_save_path + f'{use_model}_%.5f_{use_channel}_{batch_size}_entropy_{entropy_hyperparam}.txt'%(learning_rate)
    print('logging filename : ',logging_filename)
    print('save filename : ',save_filename)
    
    # exit(1)
    train_resnet_dataloader_representationlearning(save_filename=save_filename,logging_filename=logging_filename,train_dataset_list=training_fold_list,val_dataset_list=validation_fold_list,test_dataset_list=test_fold_list,
                                                batch_size = batch_size,entropy_hyperparam=entropy_hyperparam,
                                                epochs=epochs,optim=optim,loss_function=loss_function,
                                                learning_rate=learning_rate,scheduler=scheduler,warmup_iter=warmup_iter,cosine_decay_iter=cosine_decay_iter,stop_iter=stop_iter,
                                                use_channel=use_channel,class_num=class_num,classification_mode=classification_mode,aug_p=aug_p,aug_method=aug_method,
                                                first_conv=first_conv,maxpool=maxpool, layer_filters=layer_filters,block_kernel_size=block_kernel_size,block_stride_size=block_stride_size,
                                                gpu_num=gpu_num,sample_rate= 100,epoch_size = 30)