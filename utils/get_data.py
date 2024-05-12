import nibabel as nib
import numpy as np

# batch_x, batch_y = get_data(dir_list, batch_size)

def get_data(data_dir: list, batch_size: int, res_xyz: tuple, out_channels: int):
    # Load data from the given directory
    in_channel = 1
    data_x = np.zeros((batch_size, in_channel, *res_xyz))
    data_y = np.zeros((batch_size, out_channels, *res_xyz))
    for idx, case_dir in enumerate(data_dir):
        # load the input data
        input_path = case_dir["imagesTr"]
        input_img = nib.load(input_path)
        input_data = input_img.get_fdata()
        input_data = np.clip(input_data, -1024, 2976)
        input_data = (input_data + 1024) / 4000
        res_x, res_y, res_z = res_xyz
        # pad input data.shape[0] and data.shape[1] to res_x and res_y
        pad_x = (res_x - input_data.shape[0]) // 2
        pad_y = (res_y - input_data.shape[1]) // 2
        # rex_z is less than input_data.shape[2], so we need to select random start_z and end_z
        start_z = np.random.randint(0, input_data.shape[2] - res_z)
        end_z = start_z + res_z
        data_x[idx, 0, pad_x:pad_x+input_data.shape[0], pad_y:pad_y+input_data.shape[1], :] = input_data[:, :, start_z:end_z]

        label_path = case_dir["labelsTr"]
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()
        # crop and pad
        data_y[idx, :, pad_x:pad_x+input_data.shape[0], pad_y:pad_y+input_data.shape[1], :] = label_data[:, :, start_z:end_z]
        # create a one-hot label
        data_y[idx] = np.eye(out_channels)[data_y[idx].astype(int)]
        
    return data_x, data_y
            
