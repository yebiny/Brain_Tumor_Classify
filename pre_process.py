from process_utils import *
import pandas as pd

DATA_PATH='/Volumes/BPlus/dataset/miccai2021/rsna-miccai-brain-tumor-radiogenomic-classification'
TRAIN_DIR='%s/train'%DATA_PATH
TEST_DIR='%s/test'%DATA_PATH
TRAIN_LABELS='%s/train_labels.csv'%DATA_PATH

mode='train'

if mode=='train':
    DIR=TRAIN_DIR
    print('Start train mode processing')
elif mode=='test':
    DIR=TEST_DIR
    print('Start test mode processing')
else: 
    print("mode error")


patientIDs = [file for file in os.listdir(DIR) if not file.startswith("._")]
print(len(patientIDs))
img_types=['FLAIR', 'T1w', "T1wCE", "T2w"]
img_type = img_types[0]

if mode=='train':
    df = pd.read_csv(TRAIN_LABELS)
    p_label, v_label = df['BraTS21ID'], df['MGMT_value']
    p_label, v_label = np.array(p_label), np.array(v_label)
    print("* load label info " , p_label.shape, v_label.shape)

x_data, y_data = [], []
for patientID in patientIDs[:300]:
    for r in [None, 0]:
        for f in [None, 0, 1]:
            dcm_list = get_dcm_files(DIR, patientID, img_type)
            dcm_arr = load_dcm_imgs_3d(dcm_list, num_imgs=128, img_size=256, rotate=r, flip=f)
            patches = img_2_patches(dcm_arr, crop_size=32)
            x_data.append(patches)
            
            if mode=='train':
                y = [v_label[p_label==int(patientID)][0], p_label[p_label==int(patientID)][0]]
                y_data.append(y)
    print(patientID)

x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape, y_data.shape)

if mode=='train':
    np.save("x_data_300.npy", x_data)
    np.save("y_data_300.npy", y_data)
else: 
    np.save("x_test.npy", x_data)

