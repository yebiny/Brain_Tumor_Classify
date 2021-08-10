from process_utils import *
import pandas as pd

DATA_PATH='/Volumes/BPlus/dataset/miccai2021/rsna-miccai-brain-tumor-radiogenomic-classification'
TRAIN_DIR='%s/train'%DATA_PATH
TRAIN_LABELS='%s/train_labels.csv'%DATA_PATH


patientIDs = [file for file in os.listdir(TRAIN_DIR) if not file.startswith("._")]
img_types=['FLAIR', 'T1w', "T1wCE", "T2w"]
img_type = img_types[0]

df = pd.read_csv(TRAIN_LABELS)
p_label, v_label = df['BraTS21ID'], df['MGMT_value']
p_label, v_label = np.array(p_label), np.array(v_label)
print("* load label info " , p_label.shape, v_label.shape)

x_data, y_data = [], []
for patientID in patientIDs[400:]:
    for r in [None, 0]:
        for f in [None, 0, 1]:
            dcm_list = get_dcm_files(TRAIN_DIR, patientID, img_type)
            dcm_arr = load_dcm_imgs_3d(dcm_list, num_imgs=128, img_size=256, rotate=r, flip=f)
            patches = img_2_patches(dcm_arr, crop_size=32)
            y = [v_label[p_label==int(patientID)][0], p_label[p_label==int(patientID)][0]]
            x_data.append(patches)
            y_data.append(y)
    print(y)
x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape, y_data.shape)

np.save("x_data.npy", x_data)
np.save("y_data.npy", y_data)
