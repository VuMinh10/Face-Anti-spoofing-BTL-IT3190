class DefaultConfigs(object):
    seed = 666
    weight_decay = 5e-4
    momentum = 0.9
    init_lr = 0.01
    lr_epoch_1 = 0
    lr_epoch_2 = 150
    pretrained = True
    model = 'resnet18'
    
    gpus = "0"
    batch_size = 10
    norm_flag = True
    max_iter = 4000
    lambda_triplet = 2
    lambda_adreal = 0.1
    tgt_best_model_name = 'model_best.pth.tar' 
    
    # [ĐÃ ĐỔI] Huấn luyện trên OULU và MSU
    src1_data = 'oulu'
    src1_train_num_frames = 1
    src2_data = 'msu'
    src2_train_num_frames = 1
    
    # [ĐÃ ĐỔI] Test trên CASIA
    tgt_data = 'casia'
    tgt_test_num_frames = 2
    
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    logs = './logs/'

config = DefaultConfigs()
