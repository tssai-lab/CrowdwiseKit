"""
common - configurations
"""
import torch
import numpy as np
import argparse,os,sys
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epoch', type=int, metavar='N',default=50,
                    help='case')
parser.add_argument('--warmup_epoch', type=int, metavar='N',default=5,
                    help='case')   
parser.add_argument('--TM_epoch', type=int, metavar='N',default=10,
                    help='case')
parser.add_argument('--FT_TM_epoch', type=int, metavar='N',default=5,
                    help='case')
parser.add_argument('--GCN_TM_epoch', type=int, metavar='N',default=5,
                    help='case')                       
parser.add_argument('--experiment_name', type=str, help='case')
parser.add_argument('--warmup_lr', type=float, default=1e-2)
parser.add_argument('--TM_lr', type=float, default=1e-2)
parser.add_argument('--FT_TM_lr', type=float, default=1e-2) 
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--data_root', type=str,default='./data')
parser.add_argument('--output_address', type=str,default='./out/')
parser.add_argument('--expert_num', type=int, default=300)
parser.add_argument('--expert_type_num', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_rate', type=float, default=0.2)
parser.add_argument('--th', type=float, default=0.5)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--no_grad', action='store_true', default=False)
parser.add_argument('--svd_k', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--redun', type=float, default=2.0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=str(args.device)
if not os.path.isdir(args.output_address):
    os.makedirs(args.output_address)
name = args.experiment_name+'_seed'+str(args.seed)+'_noise_rate'+str(args.noise_rate)
log_file=open(args.output_address+name+".log",'a')
sys.stdout=log_file
class Config:
    data_root = args.data_root
    # training_size = 45000
    # validation_size = 5000
    # test_size = 10000

    missing_label = np.array([0]*10)
    missing = True
    
    num_classes = args.num_classes
    batch_size = args.batch_size
    left_learning_rate = args.lr
    epoch_num = args.epoch
    #########################
    expert_num = args.expert_num
    device_id = args.device
    noise_rate= args.noise_rate
    expert_type_num= args.expert_type_num
    #########################

