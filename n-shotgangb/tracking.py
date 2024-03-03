import os
import torchvision
import torch
import numpy as np
from numpy import quantile as quant
from PIL import Image


class visualizer():
    """
    Implements helper functions to save losses, logits, networks and intermediate visuals
    """
    def __init__(self, opt):
        """folder_losses = os.path.join(opt.checkpoints_dir, opt.exp_name, "losses")

        folder_networks = os.path.join(opt.checkpoints_dir, opt.exp_name, "models")
        if opt.phase == "train":
            folder_images = os.path.join(opt.checkpoints_dir, opt.exp_name, "images")
        else:
            folder_images = os.path.join(opt.checkpoints_dir, opt.exp_name, "evaluation")
        self.losses_saver = losses_saver(folder_losses, opt.continue_epoch)
        self.image_saver = image_saver(folder_images, opt.no_masks, opt.phase, opt.continue_epoch)"""
        folder_networks = os.path.join(opt.checkpoints_dir,  "models") #opt.exp_name,
        print("hello")
        self.network_saver = network_saver(folder_networks)#, opt.no_EMA)


    def save_networks(self, netG, netD, epoch):
        print("epoch=",epoch)
        self.network_saver.save(netG, netD, epoch)




class network_saver():
    def __init__(self, folder_networks):#, no_EMA):
        self.folder_networks = folder_networks
        #self.no_EMA = no_EMA
        os.makedirs(folder_networks, exist_ok=True)

    def save(self, netG, netD, epoch):
        torch.save(netG.state_dict(), os.path.join(self.folder_networks, str(epoch)+"_G.pth"))
        torch.save(netD.state_dict(), os.path.join(self.folder_networks, str(epoch)+"_D.pth"))
        #if not self.no_EMA:
        #    torch.save(netEMA.state_dict(), os.path.join(self.folder_networks, str(epoch)+"_G_EMA.pth"))
        with open(os.path.join(self.folder_networks, "latest_epoch.txt"), "w") as f:
            f.write(str(epoch))


