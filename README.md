# CDCL

This is the code repository for the paper:  
**Cross-Domain Complementary Learning with Synthetic Data for Multi-Person Part Segmentation**  
Kevin Lin, Lijuan Wang, Kun Luo, Yinpeng Chen, Zicheng Liu, Ming-Ting Sun
[[paper](https://arxiv.org/abs/1907.05193)] [[Youtube Demo Video](https://youtu.be/8QaGfdHwH48).]

![teaser](https://f5aa38c8-a-62cb3a1a-s-sites.googlegroups.com/site/kevinlin311tw/me/out3.gif)

![teaser](http://students.washington.edu/kvlin/data/cdcl_teaser.png)


## Installation instructions
Our codebase is tested on Ubuntu 16.04 LTS, CUDA 9.0, CUDNN 7.0, Tensorflow 1.12.0, Keras 2.1.1.
We suggest creating a new conda environment for setting up the relevant dependencies.

    $ conda env create -f cdcl_environment.yml

After creating the environment, you can activate the environment and continue running the demo code.  

    $ conda activate cdcl

Alternatively, we provide the Dockerfile which can be used to build a docker image with all dependencies pre-installed. You can find the `Dockerfile` in the folder [docker](https://github.com/kevinlin311tw/https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/docker).

If Anaconda and Docker do not work well on you machine, we list the required packages as below for you to manually install the dependencies.

    $ sudo apt-get -y update
    $ sudo apt-get -y install libboost-all-dev libhdf5-serial-dev libzmq3-dev libopencv-dev python-opencv git vim
    $ sudo pip install Cython scikit-image keras==2.1.1 configobj IPython
    $ sudo apt-get -y install python-tk python3-pip python3-dev python3-tk cuda-toolkit-9-0
    $ sudo pip3 install tensorflow-gpu==1.12.0
    $ sudo pip3 install keras==2.1.1 Cython scikit-image pandas zmq h5py opencv-python IPython configobj cupy-cuda90


## Fetch data
To run our code, you need to also fetch some additional files. Please run the following command:

    $ bash fetch_data.sh

This script will fetch the pretrained neural network models. In this demo script, we provide a pretrained model, which is firstly trained on COCO2014 and our multi-person synthetic data, and then fine-tuned on PASCAL-Person-Part dataset. 

This pre-trained model predcits `6` body parts in the images, and achieves `72.82%` mIOU on PASCAL-Person-Part dataset. We denote this model as `CDCL+PASCAL` in our paper.

## Run the inference code

This demo runs multi-person part segmentation on the test images. In order to run the inference code, you need to put your testing images in the folder [input](https://github.com/kevinlin311tw/https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/input). Our inference code will iterate all images in the folder, and generate the results in the folder [output](https://github.com/kevinlin311tw/https://github.com/kevinlin311tw/CDCL-human-part-segmentation/blob/master/output). 

We provide 2 different ways to use our inference code and models. Here is the example usage:

1. Single-scale inference: Please run the following script for single-scale inference

    $ python3 inference_7parts.py --scale=1

2. Multi-scale inference: You can also run the following script for multi-scale inference for more accurate results

    $ python3 inference_7parts.py --scale=1 --scale=0.5 --scale=0.75



## Training

Placeholder for now; code will be added after approval.


## Citation
If you find this code useful for your research, please consider citing the following paper:

	@article{lin2019cross,
	  Title          = {Cross-Domain Complementary Learning with Synthetic Data for Multi-Person Part Segmentation},
	  Author         = {Lin, Kevin and Wang, Lijuan and Luo, Kun and Chen, Yinpeng and Liu, Zicheng and Sun, Ming-Ting},
	  Journal      = {arXiv preprint arXiv:1907.05193},
	  Year           = {2019}
	}

