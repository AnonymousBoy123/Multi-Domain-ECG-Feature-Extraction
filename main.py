

if __name__ == "__main__":
    from sklearn.model_selection import KFold  #


    import numpy as np
    import h5py
    import torch
    import config as cfg
    from train_valid_and_test import train_valid_model, test_model

    import torch
    import torchvision

    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print('GPU:', torch.cuda.is_available())  # cuda是否可用
    print(torch.cuda.device_count())  # 返回GPU的数量
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    x = torch.rand(5, 3)
    print(x)
    import torch

    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")

    def from_mat_to_tensor(raw_data):
        # 先转换为 NumPy 数组，再转换为 float32
        Nparray = np.array(raw_data, dtype=np.float32)
        Transpose = np.transpose(Nparray)
        return Transpose


    # all the number of sbjects in the experiment
    # train one model for every subject

    # read the data
    eegname = cfg.process_data_dir + '/' + cfg.dataset_name
    eegdata = h5py.File(eegname, 'r')

    # 在读取时立即转换为 float16
    data = from_mat_to_tensor(np.array(eegdata['EEG'], dtype=np.float16))
    label = from_mat_to_tensor(np.array(eegdata['ENV'], dtype=np.float16))

    # random seed
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(2024)

    res = torch.zeros((cfg.sbnum, cfg.kfold_num))

    from sklearn.model_selection import KFold, train_test_split

    kfold = KFold(n_splits=cfg.kfold_num, shuffle=True, random_state=2024)

    for sb in range(cfg.sbnum):
        # get the data of specific subject
        eegdata = data[sb]
        eeglabel = label[sb]

        eegdata = eegdata.reshape(8 * int(360 * 128 / cfg.decision_window), cfg.decision_window, 10, 11)
        eeglabel = eeglabel.reshape(8 * int(360 * 128 / cfg.decision_window), cfg.decision_window)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(eegdata)):
            train_valid_model(eegdata[train_ids], eeglabel[train_ids], sb, fold)
            res[sb, fold] = test_model(eegdata[test_ids], eeglabel[test_ids], sb, fold)
        print("good job!")

    for sb in range(cfg.sbnum):
        print(sb)
        print(torch.mean(res[sb]))

    print("Saving results...")
    np.savetxt('result.csv', res.numpy(), delimiter=',')
    print("Save completed.")
