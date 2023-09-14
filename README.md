# UnifiedGesture: A Unified Gesture Synthesis Model for Multiple Skeletons

### [Arxiv](https://arxiv.org/abs/2309.07051) | [Demo](https://youtu.be/Ix22-ZRqSss)

<div center>
<img src="framework.png" width="550px">
</div>

## 1. Getting started

```gitignore
conda create -n UnifiedGesture python==3.7
conda activate UnifiedGesture
pip install -r requirements.txt
```

[//]: # (netron==6.8.8)

[//]: # (nvidia-cublas-cu11==11.10.3.66)

[//]: # (nvidia-cuda-nvrtc-cu11==11.7.99)

[//]: # (nvidia-cuda-runtime-cu11==11.7.99)

[//]: # (nvidia-cudnn-cu11==8.5.0.96)

[//]: # (typeguard==3.0.2)

[//]: # (typing_extensions==4.5.0)

[//]: # (protobuf==3.20.0)

[//]: # (pycocotools)

[//]: # (en-core-web-sm aubio)

## 2. Quick start

Download files such as pre-trained models from [Google Drive](TBA) or [Baidu Netdisk](TBA).

Put the pre-trained models and data:

* Diffusion model
* VQVAE
* Retargeting network
* Test data (Trinity, ZEGGS, mean file, std file, reference file etc.)

to according folders.

Download [WavLM model](https://github.com/microsoft/unilm/tree/master/wavlm) and put it to `./diffusion_latent/wavlm_cache`

```gitignore
cd ./diffusion_latent/
python sample.py --config=./configs/all_data.yml --gpu 0 --save_dir='./result_quick_start/Trinity' --audio_path="../dataset/ZEGGS/all_speech/005_Neutral_4_x_1_0.npy" --model_path='./experiments/256_seed_6_aux_model001700000_reinforce_diffusion_onlydiff_gradnorm0.1_lr1e-7_max0_seed0/ckpt/diffusion_epoch_1.pt'
```

Optional:
* If you want to use your own audio, please directly change the path of `--audio_path` to your own audio path such as `--audio_path='../dataset/Trinity/audio/Recording_006.wav'`
* You can refer to `generate_result()` in `sample.py` to generate all the files rather than only one file.

You will get the generated motion in `./diffusion_latent/result_quick_start/Trinity/` folder with the name `xxx_recon.npy`, `xxx_code.npy` and `xxx.npy`.

Then select the target skeleton and decode the primal gesture:
```gitignore
cd ../retargeting/
python demo.py --target ZEGGS --input_file "../diffusion_latent/result_quick_start/Trinity/005_Neutral_4_x_1_0_minibatch_1080_[0, 0, 0, 0, 0, 3, 0]_123456_recon.npy" --ref_path './datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy' --output_path '../result/inference/Trinity/' --cuda_device cuda:0
```
or
```gitignore
mkdir "../diffusion_latent/result_quick_start/ZEGGS/"
cp "../diffusion_latent/result_quick_start/Trinity/005_Neutral_4_x_1_0_minibatch_1080_[0, 0, 0, 0, 0, 3, 0]_123456_recon.npy" "../diffusion_latent/result_quick_start/ZEGGS/"
python demo.py --target Trinity --input_file "../diffusion_latent/result_quick_start/ZEGGS/005_Neutral_4_x_1_0_minibatch_1080_[0, 0, 0, 0, 0, 3, 0]_123456_recon.npy" --ref_path './datasets/bvh2latent/ZEGGS/065_Speech_0_x_1_0.npy' --output_path '../result/inference/Trinity/' --cuda_device cuda:0
```

And you will get `005_Neutral_4_x_1_0_minibatch_1080_[0, 0, 0, 0, 0, 3, 0]_123456_recon.bvh` in `"./result/inference/Trinity/"` folder.
You can ref [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture#2-quick-start) to use [Blender](https://www.blender.org/) to visualize the generated motion.
The results are shown below, try the output with different skeletons.



https://github.com/YoungSeng/UnifiedGesture/assets/37477030/a34d721a-3306-434a-9c44-de2af1701705



Finally the problem of foot sliding can be partially dealt with using inverse kinematics.

```gitignore
cd ./datasets/
python process_bvh.py --step IK --source_path "../../result/inference/Trinity/" --ref_bvh "./Mixamo_new_2/ZEGGS/067_Speech_2_x_1_0.bvh"
```

You will get `005_Neutral_4_x_1_0_minibatch_1080_[0, 0, 0, 0, 0, 3, 0]_123456_recon_fix.bvh` in the folder same as before.
The results are shown below, orange indicates the result of IK optimization performed on the lower body. And you can try to modify the threshold for foot contact speed to strike a balance between foot sliding and smoothness.



https://github.com/YoungSeng/UnifiedGesture/assets/37477030/0cc625e2-d049-465d-9aa7-c539528ea53a



## 3. Train your own model

TBA

### 3.1 Data preparation




### 3.2 Training retargeting network


### 3.4 Training diffusion model


### 3.5 Refinement

#### 3.5.1 Training VQVAE model

#### 3.5.2 RL training

### Acknowledgments

We are grateful to 
 * [Skeleton-Aware Networks for Deep Motion Retargeting](https://github.com/DeepMotionEditing/deep-motion-editing), 
 * [MDM: Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model), 
 * [Bailando: 3D dance generation via Actor-Critic GPT with Choreographic Memory](https://github.com/lisiyao21/Bailando), and 
 * [Edge: editable dance generation from music](https://github.com/Stanford-TML/EDGE)

for making their code publicly available, which helped significantly in this work.

### Citation

If you find this work useful, please cite the paper with the following bibtex:
```
@inproceedings{yang2023UnifiedGesture,
  title={UnifiedGesture: A Unified Gesture Synthesis Model for Multiple Skeletons},
  author={Sicheng Yang, Zilin Wang, Zhiyong Wu, Minglei Li, Zhensong Zhang, Qiaochu Huang, Lei Hao, Songcen Xu, Xiaofei Wu, changpeng yang, Zonghong Dai},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023},
  doi={https://doi.org/10.1145/3581783.3612503}
}
```

If you have any questions, please contact us at [yangsc21@mails.tsinghua.edu.cn](yangsc21@mails.tsinghua.edu.cn)
or [wangzl21@mails.tsinghua.edu.cn](wangzl21@mails.tsinghua.edu.cn).
