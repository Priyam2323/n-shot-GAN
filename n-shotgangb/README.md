# n-Shot GAN

## 0. Data
The datasets used in the paper can be found at [link][(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)]. 
 

## 1. Description
The repository is motivated by [link][ (https://github.com/odegeasslbc/FastGAN-pytorch)] and extended for the task 
The code is structured as follows:

* train.py: this is the main entry of the code, execute this file to train the model, the intermediate results will be automatically saved periodically into a folder "train_results" and encoder weights will be saved in "checkpoints folder"

* eval.py: generates images from a trained generator into a folder, which can be used to calculate FID score.
* tr_encoder.py : this file is used to train the encoder in order to carve ou the features of the image created by eval.py

*test_ad.py : this folder is the test script to find final anomaly score and graphs.

## 2. How to run
Place all your training images in a folder, and simply call
```
python train.py --path /path/to/RGB-image-folder --iter 10000  #iterations can change
```
You can also see all the training options by:
```
python train.py --help
```
The code will automatically create a new folder (you have to specify the name of the folder using --name option);by default : test1; to store the trained checkpoints and intermediate synthesis results.

Once finish training, we can generate 100 images (or as many as you want) by:
```
cd ./train_results/name_of_your_training/
python eval.py --n_sample 100 
```

cd ../..
python tr_encoder.py t--path train_results/test1/eval_10000/img --iter 10
python test_ad.py  #mention the path in test_ad.py file of test dataset



