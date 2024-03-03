import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torchvision.datasets import ImageFolder

import argparse
import random
from tqdm import tqdm
from torch.utils.model_zoo import tqdm
import tracking
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
from encoder import Encoder, DiscriminatorD

from skimage.metrics import structural_similarity as compare_ssim
from skimage import io, color

policy = 'color,translation'
import lpips
#percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True
        

def test(args):

    data_root = ["small_cov/norm","pnemonia_lungs/kaggle_pneu"]
    with open("score_pneu_500.csv", "w") as f:
            f.write("label,img_distance,anomaly_score,z_distance\n")
    j=0  
    for i in data_root:
        j=j+1
        #total_iterations = args.iter
        checkpoint = args.ckpt
        batch_size = args.batch_size
        im_size = args.im_size
        ndf = 64
        ngf = 64
        nz = 256
        nlr = 0.0002
        nbeta1 = 0.5
        use_cuda = True
        multi_gpu = True
        dataloader_workers = 8
        #current_iteration = args.start_iter
        #save_interval = 100
        #saved_model_folder, saved_image_folder = get_dir(args)
        
    
    
       
        pipeline = [transforms.Resize([args.im_size]*2),
                transforms.RandomHorizontalFlip()]
        if args.channels == 1:
            pipeline.append(transforms.Grayscale())
        pipeline.extend([transforms.ToTensor(),
                         transforms.Normalize([0.5]*args.channels, [0.5]*args.channels)])
        transform = transforms.Compose(pipeline)
        dataset = ImageFolder(i, transform=transform)#opt.test_root,
        print("dataset = ",i)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
      
        net_ig = Generator( ngf=ngf, nz=nz, nc=3, im_size=args.im_size)
        #net_ig.to(device) 
        
        ckpt = f"train_results/test1/models/all_10000.pth"
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        # Remove prefix `module`.
        checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])
        #print('load checkpoint success, epoch %d'%epoch)
        print("checkpoint load success")
        net_ig.to(device)
        
        print("lebgth=",len(dataloader))
        print("root=",data_root)
        encoder=Encoder(args)
        encoder.load_state_dict(torch.load("checkpoints/encodernpneu"))
        
        discriminatorD=DiscriminatorD(args)
        optimizerE = optim.Adam(encoder.parameters(), lr=nlr, betas=(nbeta1, 0.999))
        discriminatorD.to(device)
        encoder.to(device)
        
        criterion = nn.MSELoss()
        kappa=1
        
        print("out of for loop")
        
        for (img) in tqdm(dataloader):
          label=j
    
          real_img = img.to(device)
          #print("real_img =",real_img.shape)
          real_z = encoder(real_img)
          #print("real_z =",real_z.shape)
          #real_z=real_z.reshape(1,64,1,1)
          fake_img=net_ig(real_z)[0]
          fake_z = encoder(fake_img)
      
          real_feature = discriminatorD.forward_features(real_img)
          fake_feature = discriminatorD.forward_features(fake_img)
      
          # Scores for anomaly detection
          img_distance = criterion(fake_img, real_img)
          loss_feature = criterion(fake_feature, real_feature)
          anomaly_score = (img_distance) + kappa * loss_feature
      
          z_distance = criterion(fake_z, real_z)
          
          real_img_np = real_img.squeeze().cpu().detach().numpy()
          fake_img_np = fake_img.squeeze().cpu().detach().numpy()
      
          # Convert images to grayscale if needed
          real_img_gray = real_img_np.mean(axis=-1) if real_img_np.shape[-1] == 3 else real_img_np
          fake_img_gray = fake_img_np.mean(axis=-1) if fake_img_np.shape[-1] == 3 else fake_img_np
          data_range = real_img_gray.max() - real_img_gray.min()
          # Calculate SSIM
          #ssim_value, _ = compare_ssim(real_img_gray, fake_img_gray, full=True,win_size=3,data_range=data_range)

          #anomaly_ssim = (ssim_value) + kappa * loss_feature
          #anomaly_ssim=(ssim_value-1)/2
        
      
          with open("score_pneu_500.csv", "a") as f:
              f.write(f"{label},{img_distance},"f"{anomaly_score},{z_distance}\n")
      
    
   
   
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')
    #parser.add_argument("test_root", type=str,help="root name of your dataset in test mode")
    parser.add_argument('--path', type=str, default='../montogomery', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    #parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    #parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument("--channels", type=int, default=3,help="number of image channels (If set to 1, convert image to grayscale)")

    args = parser.parse_args()
    print(args)

    test(args)
