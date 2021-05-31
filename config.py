import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--n_labels', type=int, default=2,help='number of classes') # 分割肝脏则置为2（二类分割），分割肝脏和肿瘤则置为3（三类分割）
parser.add_argument('--upper', type=int, default=1000, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--expand_slice', type=int, default=20, help='')
parser.add_argument('--min_slices', type=int, default=48, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')

# data in/out and dataset
parser.add_argument('--weight', type=str, default=None, help='model init weight')
parser.add_argument("--save_model", default='/content/gdrive/Shareddrives/课程实验/runs',
        help="Whether to save the trained model.")
parser.add_argument("--train_image_dir", default = '/content/gdrive/Shareddrives/课程实验/datasets/ribfrac-train-images', required=False,
        help="The training image nii directory.")
parser.add_argument("--train_label_dir", default = '/content/gdrive/Shareddrives/课程实验/datasets/ribfrac-train-labels', required=False,
        help="The training label nii directory.")
parser.add_argument("--val_image_dir", default = '/content/gdrive/Shareddrives/课程实验/datasets/ribfrac-val-images', required=False,
        help="The validation image nii directory.")
parser.add_argument("--val_label_dir", default = '/content/gdrive/Shareddrives/课程实验/datasets/ribfrac-val-labels', required=False,
        help="The validation label nii directory.")
parser.add_argument('--save',default='UNet',help='save path of trained model')
parser.add_argument('--batch_size', type=int, default=2,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=64)
parser.add_argument('--val_crop_max_size', type=int, default=96)

# test
parser.add_argument('--test_cut_size', type=int, default=48, help='size of sliding window')
parser.add_argument('--test_cut_stride', type=int, default=24, help='stride of sliding window')
parser.add_argument('--postprocess', type=bool, default=False, help='post process')


args = parser.parse_args()


