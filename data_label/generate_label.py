import os
import json
import glob
from sklearn.model_selection import train_test_split

# 1. TRỎ ĐƯỜNG DẪN ĐỌC ẢNH TỪ Ổ ĐĨA CD (INPUT)
data_dir = '/kaggle/input/notebooks/vuminh10/btl-h-c-m-y-ti-n-x-l-d-li-u/BTL-IT3190/data/'

# 2. CHỈ ĐỊNH LƯU FILE JSON RA Ổ NHÁP (WORKING)
save_base_dir = '/kaggle/working/BTL-IT3190/data_label/'

def save_json_files(dataset_name, train_data, valid_data, test_data, all_data, real_data, fake_data):
    save_dir = os.path.join(save_base_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'train_label.json'), 'w') as f: json.dump(train_data, f, indent=4)
    if valid_data is not None:
        with open(os.path.join(save_dir, 'valid_label.json'), 'w') as f: json.dump(valid_data, f, indent=4)
    with open(os.path.join(save_dir, 'test_label.json'), 'w') as f: json.dump(test_data, f, indent=4)
    with open(os.path.join(save_dir, 'all_label.json'), 'w') as f: json.dump(all_data, f, indent=4)
    with open(os.path.join(save_dir, 'real_label.json'), 'w') as f: json.dump(real_data, f, indent=4)
    with open(os.path.join(save_dir, 'fake_label.json'), 'w') as f: json.dump(fake_data, f, indent=4)

def msu_process():
    train_json, test_json, all_json, real_json, fake_json = [], [], [], [], []
    dataset_path = os.path.join(data_dir, 'msu/')
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True) + glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()

    for path in path_list:
        label = 1 if '/real/' in path.lower() else 0
        item = {'photo_path': path, 'photo_label': label}
        all_json.append(item)
        (real_json if label == 1 else fake_json).append(item)
        if '/train/' in path.lower(): train_json.append(item)
        else: test_json.append(item)

    print(f'MSU -> Total: {len(path_list)} | Train: {len(train_json)} | Test: {len(test_json)}')
    save_json_files('msu', train_json, None, test_json, all_json, real_json, fake_json)

def casia_process():
    train_json, test_json, all_json, real_json, fake_json = [], [], [], [], []
    dataset_path = os.path.join(data_dir, 'casia/')
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True) + glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()

    for path in path_list:
        filename = path.split('/')[-1].lower()
        label = 1 if ('real' in filename or flag_is_real_in_casia(path)) else 0
        item = {'photo_path': path, 'photo_label': label}
        all_json.append(item)
        (real_json if label == 1 else fake_json).append(item)
        if '/train/' in path.lower(): train_json.append(item)
        else: test_json.append(item)

    print(f'CASIA -> Total: {len(path_list)} | Train: {len(train_json)} | Test: {len(test_json)}')
    save_json_files('casia', train_json, None, test_json, all_json, real_json, fake_json)

def flag_is_real_in_casia(path):
    try:
        parts = path.split('/')[-1].split('_')
        if len(parts) >= 4 and parts[3] == 'real.jpg': return True
    except: pass
    return False

def oulu_process():
    train_json, valid_json, test_json, all_json, real_json, fake_json = [], [], [], [], [], []
    dataset_path = os.path.join(data_dir, 'oulu/')
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True) + glob.glob(dataset_path + '**/*.png', recursive=True)
    path_list.sort()

    for path in path_list:
        label = 1 if '/true/' in path.lower() else 0
        item = {'photo_path': path, 'photo_label': label}
        all_json.append(item)
        (real_json if label == 1 else fake_json).append(item)

    if all_json:
        train_json, temp_test = train_test_split(all_json, test_size=0.3, random_state=42)
        valid_json, test_json = train_test_split(temp_test, test_size=0.5, random_state=42)

    print(f'OULU -> Total: {len(path_list)} | Train: {len(train_json)} | Test: {len(test_json)}')
    save_json_files('oulu', train_json, valid_json, test_json, all_json, real_json, fake_json)

if __name__ == "__main__":
    msu_process()
    casia_process()
    oulu_process()
