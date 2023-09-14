'''
Download from  https://pan.baidu.com/s/1z1cQiqLUgjfxlWoajIPr0g (ye1q)

cd ./deep-motion-editing/retargeting/datasets/
tar -jxvf test_set.tar.bz2
cd ..
python demo.py

python test.py
Collecting test error...
Intra-retargeting error: 0.00047083793936272735
Cross-retargeting error: 0.002252287446935629
Evaluation finished!

Download from https://pan.baidu.com/s/1PM0maLNCJxBsR9RIv18cTA (4rgv)
cd ./datasets/
tar -jxvf training_set.tar.bz2

# git filter-branch --tree-filter 'rm -f ./deep-motion-editing/retargeting/datasets/training_set.tar.bz2' --tag-name-filter cat -- --all


Download from https://www.mixamo.com/ Silly Dancing and Hip Hop Dancing
# 确保blender可执行程序所在目录添加到系统环境变量path, https://zhuanlan.zhihu.com/p/525475118
# windows
blender -b -P fbx2bvh.py

cd ..
python ./datasets/preprocess.py
((135, 75), (116, 75))
python train.py --save_dir=./training/
# windows
pip install tensorboard
tensorboard --logdir=./logs/
'''

import pdb

import torch
from vector_quantize_pytorch import ResidualVQ

# python main.py
residual_vq = ResidualVQ(
    dim = 7 * 16,
    codebook_size = 512,
    num_quantizers = 1,
    kmeans_init = True,   # set to True
    kmeans_iters = 10     # number of kmeans iterations to calculate the centroids for the codebook on init
)

x = torch.randn(2, 11, 7 * 16)
quantized, indices, commit_loss = residual_vq(x)

pdb.set_trace()
