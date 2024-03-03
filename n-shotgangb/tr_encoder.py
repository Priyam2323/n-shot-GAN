import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from torchvision.utils import save_image

import argparse
import random
from tqdm import tqdm
import tracking
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
from encoder import Encoder, DiscriminatorD
import os 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
policy = 'color,translation'
#import lpips 
#percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


#torch.backends.cudnn.benchmark = True



def train(args):

    data_root = args.path
    total_iterations = args.iter
    #checkpoint = args.ckpt
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
    current_iteration = args.start_iter
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    elosses=[]
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
   
    dataset = ImageFolder(root=data_root, transform=trans)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)#,
                     # sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    #optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    #optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
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
    
    
    encoder=Encoder(args)
    discriminatorD=DiscriminatorD(args)
    
    optimizerE = optim.Adam(encoder.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    discriminatorD.to(device)
    encoder.to(device)

    
    
    os.makedirs("images_e", exist_ok=True)
    
    criterion = nn.MSELoss()
    kappa=1
    
    batches_done=0
    
    
    #for iteration in tqdm(range(current_iteration, total_iterations+1)):
    for epoch in range(args.iter):
      for i, (imgs) in enumerate(dataloader):
        
        #real_image = next(dataloader)
        real_imgs = imgs.to(device)
        real_imgs = real_imgs.to(device)
        optimizerE.zero_grad()
        
        #print("real_img =",real_img.shape)
        z = encoder(real_imgs)
        fake_imgs = net_ig(z)[0]
        #fake_imgs = F.interpolate(fake_imgs, 512)
        
        
    
        fake_feat=discriminatorD.forward_features(fake_imgs)
        real_feat=discriminatorD.forward_features(real_imgs)

        
        # Compute the loss (example: Mean Squared Error)
        loss_features = criterion(real_feat, fake_feat)
    
        loss_imgs = criterion(fake_imgs, real_imgs)
        #loss_features = criterion(fake_feat, real_feat)
        e_loss = loss_imgs + kappa * loss_features
        elosses.append(e_loss.item())
        
        e_loss.backward()
        optimizerE.step()
        padding_epoch = len(str(args.iter))
        padding_i = len(str(len(dataloader)))
        # Output training log every n_critic steps
        if i % args.n_critic == 0:
           
            print(f"[Epoch {epoch:{padding_epoch}}/{args.iter}] "
                  f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                  f"[E loss: {e_loss.item():3f}]")

            if batches_done % args.sample_interval == 0:
                #fake_imgs=fake_imgs.reshape(5,3,64,64)
                fake_z = encoder(fake_imgs)
                #print("fake_z shape = ",fake_imgs.shape)
                #fake_z = fake_z.reshape(5, 64, 1, 1)  # Reshape based on actual batch size and feature dimensions
        
                #fake_z = fake_z.reshape(5, 3, 1, 1)
                reconfiguration_imgs = net_ig(fake_z)[0]
                save_image(reconfiguration_imgs.data[:25],
                           f"images_e/{batches_done:06}.png",
                           nrow=5, normalize=True)

            batches_done += args.n_critic
    # Plot the loss curves
    
    plt.figure(figsize=(10, 5))
    plt.plot(elosses, label='Encoder Loss', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Loss') 
    plt.title('Encoder Losses Over Training')
    plt.legend()
    plt.ylim(0,0.25) 
    plt.savefig('your_graphcovidx.jpg', format='jpg')
    #plt.show()
    plt.close()
    print("Training completed.")
    torch.save(encoder.state_dict(), "checkpoints/encodernpneu") 
        









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=4, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    #parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument("--n_critic", type=int, default=5,help="number of training steps for discriminator per iter")
    parser.add_argument("--sample_interval", type=int, default=100,help="interval betwen image samples")
    parser.add_argument("--channels", type=int, default=3,help="number of image channels (If set to 1, convert image to grayscale)")
    args = parser.parse_args()
    print(args)

    train(args)
