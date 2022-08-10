from . import *




def train_deepsleepnet_dataloader(save_filename,logging_filename,train_dataset_list,val_dataset_list,test_dataset_list,batch_size = 512,entropy_hyperparam=0.,
                                                epochs=100,optim='Adam',loss_function='CE',
                                                learning_rate=0.001,scheduler=None,warmup_iter=20,cosine_decay_iter=40,stop_iter=10,
                                                use_channel=[0,1],class_num=6,classification_mode='6class',
                                                gpu_num=0,sample_rate= 100,epoch_size = 30):
    # cpu processor num
    cpu_num = multiprocessing.cpu_count()
    

    #dataload Training Dataset
    train_dataset = Sleep_Dataset_withPath_sleepEDF(dataset_list=train_dataset_list,class_num=class_num,
    use_channel=use_channel,use_cuda = True,classification_mode=classification_mode)
    train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=True, num_workers=(cpu_num//4))

    # calculate weight from training dataset (for "Class Balanced Weight")
    weights,count = make_weights_for_balanced_classes(train_dataset.signals_files_path)
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
    model = DeepSleepNet_CNN(in_channel=len(use_channel),out_channel=class_num,layer=[64,128,128,128],activation='relu',sample_rate = sample_rate,dropout_p=0.5)
                                   
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
    if loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif loss_function == 'CEW':
        samples_per_cls = count / np.sum(count)
        no_of_classes = class_num
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    elif loss_function == 'FL':
        loss_fn = FocalLoss(gamma=2).to(device)
    elif loss_function == 'CBL':
        samples_per_cls = count / np.sum(count)
        loss_fn = CB_loss(samples_per_cls=samples_per_cls, no_of_classes=class_num, loss_type='focal', beta=0.9999,
                          gamma=2.0)
    # loss_fn = FocalLoss(gamma=2).to(device)
    if entropy_hyperparam > 0.:
        loss_fn2 = Entropy()
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
    elif scheduler == 'StepLR':
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [11, 21], gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif scheduler == 'Reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.5, patience=10,
                                                           min_lr=1e-6)
    elif scheduler == 'Cosine':
        print('Cosine Scheduler')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max=epochs)


    # scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=.5)
    # loss의 값이 최소가 되도록 하며, 50번 동안 loss의 값이 감소가 되지 않을 경우 factor값 만큼
    # learning_rate의 값을 줄이고, 최저 1e-6까지 줄어들 수 있게 설정
    
    best_accuracy = 0.
    stop_count = 0
    best_test_accuracy = 0.
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
                
                # norm = 0
                # for parameter in model.parameters():
                #     norm += torch.norm(parameter, p=norm_square)

                loss = loss_fn(pred, batch_label) # + beta * norm
                if entropy_hyperparam > 0.:
                    if check_loss == False: # Only once access!
                        print('Using Entropy loss for training!')
                        check_loss = True
                    loss2 = loss_fn2(pred)
                    loss = loss + entropy_hyperparam * loss2
                
                _, predict = torch.max(pred, 1)
                
                check_count = (predict == batch_label).sum().item()

                train_total_loss += loss.item()

                train_total_count += check_count
                train_total_data += len(batch_signal)
                loss.backward()
                optimizer.step()

                accuracy = train_total_count / train_total_data
                tepoch.set_postfix(loss=train_total_loss/(index+1),accuracy=100.*accuracy)
                
        train_total_loss /= index
        train_accuracy = train_total_count / train_total_data * 100

        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                    % (epoch + 1, epochs, time.time() - start_time, train_total_loss,
                        train_total_count, train_total_data, train_accuracy)
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

                    loss = loss_fn(pred, batch_label)

                    # acc
                    _, predict = torch.max(pred, 1)
                    check_count = (predict == batch_label).sum().item()

                    val_total_loss += loss.item()
                    val_total_count += check_count
                    val_total_data += len(batch_signal)
                    accuracy = val_total_count / val_total_data
                    tepoch.set_postfix(loss=val_total_loss/(index+1),accuracy=100.*accuracy)

        val_total_loss /= index
        val_accuracy = val_total_count / val_total_data * 100

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                    % (epoch + 1, epochs, time.time() - start_time, val_total_loss,
                        val_total_count, val_total_data, val_accuracy)
        # sys.stdout.write(output_str)
        check_file.write(output_str)
        
        # scheduler.step(float(val_total_loss))
        # scheduler.step(epoch)
        if epoch == 0:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_file = save_filename
            torch.save(model.state_dict(), save_file)
            stop_count = 0
            test_total_count = 0
            test_total_data = 0
            # check validation dataset
            start_time = time.time()
            model.eval()

            with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
                for index,(batch_signal, batch_label) in enumerate(tepoch):
                    batch_signal = batch_signal.to(device)
                    batch_label = batch_label.long().to(device)

                    with torch.no_grad():
                        pred = model(batch_signal)

                        loss = loss_fn(pred, batch_label)

                        # acc
                        _, predict = torch.max(pred, 1)
                        check_count = (predict == batch_label).sum().item()

                        test_total_count += check_count
                        test_total_data += len(batch_signal)
                        accuracy = test_total_count / test_total_data
                        tepoch.set_postfix(accuracy=100.*accuracy)


            test_accuracy = test_total_count / test_total_data * 100
            best_test_accuracy = test_accuracy
            output_str = 'test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                        % (epoch + 1, epochs, time.time() - start_time,
                            test_total_count, test_total_data, test_accuracy)
            # sys.stdout.write(output_str)
            check_file.write(output_str)
        else:
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                save_file = save_filename
                torch.save(model.state_dict(), save_file)
                stop_count = 0
                test_total_count = 0
                test_total_data = 0
                # check validation dataset
                start_time = time.time()
                model.eval()

                with tqdm(test_dataloader,desc='Test',unit='batch') as tepoch:
                    for index,(batch_signal, batch_label) in enumerate(tepoch):
                        batch_signal = batch_signal.to(device)
                        batch_label = batch_label.long().to(device)

                        with torch.no_grad():
                            pred = model(batch_signal)

                            loss = loss_fn(pred, batch_label)

                            # acc
                            _, predict = torch.max(pred, 1)
                            check_count = (predict == batch_label).sum().item()

                            test_total_count += check_count
                            test_total_data += len(batch_signal)
                            accuracy = test_total_count / test_total_data
                            tepoch.set_postfix(accuracy=100.*accuracy)


                test_accuracy = test_total_count / test_total_data * 100
                best_test_accuracy = test_accuracy
                output_str = 'test dataset : %d/%d epochs spend time : %.4f sec  / correct : %d/%d -> %.4f%%\n' \
                            % (epoch + 1, epochs, time.time() - start_time,
                                test_total_count, test_total_data, test_accuracy)
                # sys.stdout.write(output_str)
                check_file.write(output_str)
            else:
                stop_count += 1
        if stop_count > stop_iter:
            print('Early Stopping')
            break
        
        output_str = 'best epoch : %d/%d / test accuracy : %f%%\n' \
                    % (best_epoch + 1, epochs, best_test_accuracy)
        sys.stdout.write(output_str)
        print('=' * 30)

    output_str = 'best epoch : %d/%d / test accuracy : %f%%\n' \
                 % (best_epoch + 1, epochs, best_test_accuracy)
    sys.stdout.write(output_str)
    check_file.write(output_str)
    print('=' * 30)

    check_file.close()





def training_deepsleepnet_dataloader(use_dataset='sleep_edf',total_train_percent = 1.,train_percent=0.8,val_percent=0.1,test_percent=0.1,random_seed=2,use_channel=[0,1],entropy_hyperparam=0.,classification_mode='6class',gpu_num=[0]):
    
    if use_dataset == 'sleep_edf':
        signals_path = '/home/eslab/dataset/sleep_edf/origin_npy/remove_wake_version0/each/'

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
    batch_size = 512
    warmup_iter=10
    cosine_decay_iter=10
    learning_rate = 10**-4
    stop_iter = 10
    loss_function = 'CE' # CEs
    optim= 'Adam'
    scheduler = 'Cosine' # 'WarmUp_restart'    
    
    
    print(f'class num = {class_num}')
    model_save_path = f'/data/hdd3/git/DeepSleepNet_pytorch/saved_model/{use_dataset}/{classification_mode}/single_epoch_models_{round(train_percent,2)}_{round(val_percent,2)}_{round(test_percent,2)}/random_seed_{random_seed}/'
    logging_save_path = f'/data/hdd3/git/DeepSleepNet_pytorch/log/{use_dataset}/{classification_mode}/single_epoch_models_{round(train_percent,2)}_{round(val_percent,2)}_{round(test_percent,2)}/random_seed_{random_seed}/'
    # model_save_path = '/home/eslab/kdy/git/Sleep_pytorch/saved_model/seoulDataset/single_epoch_models/'
    # logging_save_path = '/home/eslab/kdy/git/Sleep_pytorch/log/seoulDataset/single_epoch_models/'

    os.makedirs(model_save_path,exist_ok=True)
    os.makedirs(logging_save_path,exist_ok=True)
    
    save_filename = model_save_path + f'DeepSleepNet_%.5f_{use_channel}_entropy_{entropy_hyperparam}.pth'%(learning_rate)
    
    logging_filename = logging_save_path + f'DeepSleepNet_%.5f_{use_channel}_entropy_{entropy_hyperparam}.txt'%(learning_rate)
    print('logging filename : ',logging_filename)
    print('save filename : ',save_filename)
    
    # exit(1)
    train_deepsleepnet_dataloader(save_filename=save_filename,logging_filename=logging_filename,train_dataset_list=training_fold_list,val_dataset_list=validation_fold_list,test_dataset_list=test_fold_list,
                                                batch_size = batch_size,entropy_hyperparam=entropy_hyperparam,
                                                epochs=epochs,optim=optim,loss_function=loss_function,
                                                learning_rate=learning_rate,scheduler=scheduler,warmup_iter=warmup_iter,cosine_decay_iter=cosine_decay_iter,stop_iter=stop_iter,
                                                use_channel=use_channel,class_num=class_num,classification_mode='6class',
                                                gpu_num=gpu_num,sample_rate= 100,epoch_size = 30)