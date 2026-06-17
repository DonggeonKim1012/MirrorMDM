# Mirror MDM: Enhancing directional awarness in diffusion text to motion


[![arXiv](https://img.shields.io/badge/arXiv-<2209.14916>-<COLOR>.svg)](https://arxiv.org/abs/2209.14916)
<a href="https://replicate.com/arielreplicate/motion_diffusion_model"><img src="https://replicate.com/arielreplicate/motion_diffusion_model/badge"></a>

The followup PyTorch implementation of the paper [**"Human Motion Diffusion Model"**](https://arxiv.org/abs/2209.14916).

Please visit their [**webpage**](https://guytevet.github.io/mdm-page/) for more details.

If you'd like access to our paper(Mirror MDM), please contact via gundong@khu.ac.kr

## Bibtex

If you find this code useful in your research, please cite:

```
MDM:

@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}

DiP and CLoSD:

@article{tevet2024closd,
  title={CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character control},
  author={Tevet, Guy and Raab, Sigal and Cohan, Setareh and Reda, Daniele and Luo, Zhengyi and Peng, Xue Bin and Bermano, Amit H and van de Panne, Michiel},
  journal={arXiv preprint arXiv:2410.03441},
  year={2024}
}
```

No citation is required for the MirrorMDM paper.

## Getting started

This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

<details>
  <summary><b>Text to Motion</b></summary>

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```
</details>


### 2. Get data

**Text to Motion** 

[Download HumanML3D](https://drive.google.com/drive/folders/1OZrTlAGRvLjXhXwnRiOC-oxYry1vf-Uu?usp=drive_link)

Or, alternatively, parse the data yourself according to the original instructions:


<details>
  <summary><b>Original Text to Motion instructions</b></summary>

There are two paths to get the data:

(a) **Go the easy way if** you just want to generate text-to-motion (excluding editing which does require motion capture data)

(b) **Get full data** to train and evaluate the model.


#### a. The easy way (text only)

**HumanML3D** - Clone HumanML3D, then copy the data dir to our repository:

```shell
cd ..
git clone https://github.com/EricGuo5513/HumanML3D.git
unzip ./HumanML3D/HumanML3D/texts.zip -d ./HumanML3D/HumanML3D/
cp -r HumanML3D/HumanML3D motion-diffusion-model/dataset/HumanML3D
cd motion-diffusion-model
```


#### b. Full data (text + motion capture)

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) (no processing needed this time) and the place result in `./dataset/KIT-ML`
</details>


### 3. Download the pretrained models

Download the model(s) you wish to use, then unzip and place them in `./save/`. 

<details>
  <summary><b>Text to Motion</b></summary>

**You need only the first one.** 

**HumanML3D**

[NEW!] [humanml_trans_dec_512_bert-50steps](https://drive.google.com/file/d/1z5IW5Qa9u9UdkckKylkcSXCwIYgLPhIC/view?usp=sharing) - Runs 20X faster with improved precision!

[NEW!] [humanml-encoder-512-50steps](https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view?usp=sharing) - Runs 20X faster with comparable performance!

[humanml-encoder-512](https://drive.google.com/file/d/1PE0PK8e5a5j-7-Xhs5YET5U5pGh0c821/view?usp=sharing) (best model used in the paper)

[humanml-decoder-512](https://drive.google.com/file/d/1q3soLadvVh7kJuJPd2cegMNY2xVuVudj/view?usp=sharing)

[humanml-decoder-with-emb-512](https://drive.google.com/file/d/1GnsW0K3UjuOkNkAWmjrGIUmeDDZrmPE5/view?usp=sharing)

**KIT**

[kit-encoder-512](https://drive.google.com/file/d/1SHCRcE0es31vkJMLGf9dyLe7YsWj7pNL/view?usp=sharing)

</details>


## Training
<details>
  <summary><b>Original MDM</b></summary>
```shell
python -m train.train_mdm \
    --save_dir save/humanml_trans_enc_512 \
    --dataset humanml \
    --diffusion_steps 1000 --noise_schedule cosine \
    --arch trans_enc --layers 8 --latent_dim 512
```
</details>
<details>
  <summary><b>InfoNCE</b></summary>
```shell
python -m train.train_mdm_infonce \
    --save_dir save/humanml_infonce_baseline \
    --dataset humanml \
    --diffusion_steps 1000 --noise_schedule cosine \
    --lambda_infonce 0.1 --infonce_temperature 0.07 \
    --num_steps 200000 \
    --train_platform_type WandBPlatform --wandb_project mdm_infonce
```
</details>
<details>
  <summary><b>Ours</b></summary>
```shell
python -m train.train_mdm_contrastive_v2 \
  --save_dir save/humanml_contrastive_1000step \
  --dataset humanml \
  --diffusion_steps 1000 --noise_schedule cosine \
  --flipped_motion_dir dataset/HumanML3D_flipped/new_joint_vecs \
  --lambda_contrastive 0.1 --contrastive_margin 0.05 \
  --lambda_warmup_steps 10000 \
  --num_steps 200000 \
  --train_platform_type WandBPlatform --wandb_project mdm_contrastive
```
</details>

## Motion Synthesis
<details>
  <summary><b>Text to Motion</b></summary>

### Generate from test set prompts

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your text file

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/example_text_prompts.txt
```

### Generate a single prompt

```shell
python -m sample.generate --model_path ./save/humanml_contrastive_v2_50/model000750000.pt --text_prompt "a person walks while touching something with his right hand."
```
</details>

## Evaluate

<details>
  <summary><b>Original paper evaluation</b></summary>


```shell
python -m eval.eval_humanml \
    --model_path save/humanml_trans_enc_512/humanml_trans_enc_512/model000200000.pt \
    --eval_mode wo_mm --guidance_param 2.5

# InfoNCE 1000-step
python -m eval.eval_humanml \
    --model_path save/humanml_infonce_baseline/model000200000.pt \
    --eval_mode wo_mm --guidance_param 2.5

# Contrastive 1000-step
python -m eval.eval_humanml \
    --model_path save/humanml_contrastive_1000step/model000200000.pt \
    --eval_mode wo_mm --guidance_param 2.5
```

</details>
<details>
  <summary><b>AOur preset prompts</b></summary>

```shell
python -m eval.eval_lr_prompts     --model_path save/humanml_trans_enc_512/humanml_trans_enc_512/model000200000.pt     --n_frames 100     --repeats 3
python -m eval.eval_lr_prompts     --model_path save/humanml_infonce_baseline/model000200000.pt     --n_frames 100     --repeats 3
python -m eval.eval_lr_prompts     --model_path save/humanml_contrastive_1000step/model000200000.pt     --n_frames 100     --repeats 3
```
</details>

## Acknowledgments

This code is solely based on the original MDM repository. Thank you for open sourcing such an exciting project.

## License
This code was distributed under an [MIT LICENSE](LICENSE).

We claim no ownerhsip of the code. Follow the MIT license to refactor it.

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
