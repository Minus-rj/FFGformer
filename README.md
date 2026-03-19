# FFGformer
Evaluation metrics
[UIQM](https://github.com/xahidbuffon/FUnIE-GAN/blob/master/Evaluation/uqim_utils.py), [UCIQE](https://github.com/Duanlab123/MetaUE/blob/main/utils/evalution.py),  **PSNR**, **SSIM**, **LPIPS**, **DISTS**,  **NIQE** and  **LIQE** ([see Evaluation]([/Evaluation/](https://github.com/chaofengc/IQA-PyTorch)))

## Dataset

[UIEB](https://li-chongyi.github.io/proj_benchmark.html)resplited by [URanker](https://github.com/RQ-Wu/UnderwaterRanker?tab=readme-ov-file#underwater-ranker-learn-which-is-better-and-how-to-be-better-aaai-2023-oral-presentation),[TURBID](http://amandaduarte.com.br/turbid/)resplited by [DGD-cGAN](https://github.com/SalPGS/DGD-cGAN),

[UFO-120](https://www.kaggle.com/datasets/lmslms/ufo-120),[Seathru](https://www.kaggle.com/datasets/colorlabeilat/seathru-dataset),[U45](https://github.com/IPNUISTlegal/underwater-test-dataset-U45-)

## Comparison Methods

1)Pixel-adjustment based method:[WWPF](https://github.com/Li-Chongyi/WWPF_code),[WFAC](https://www.researchgate.net/publication/386508762_2024WFAC),

2)Imaging model based method:[UNTV](https://github.com/Hou-Guojia/UNTV),[ISCP](https://github.com/Hou-Guojia/ICSP)

3)CNN based method:[Water-Net](https://github.com/Li-Chongyi/Water-Net_Code),[PUIE-Net](https://github.com/zhenqifu/PUIE-Net)

4)GAN based method:[DGD-cGAN](https://github.com/SalPGS/DGD-cGAN) [PUGAN](https://github.com/rmcong/PUGAN_TIP2023?tab=readme-ov-file)

5)Transformer based method:[URSCT-SESR](https://github.com/TingdiRen/URSCT-SESR),[UVZ](https://github.com/WindySprint/UVZ),[PhaseFormer](https://github.com/Mdraqibkhan/Phaseformer)

## Application Experients
Low-light image enhancement [LOL-v1](https://daooshee.github.io/BMVC2018website/)

salient object detection[U-2-Net](https://github.com/xuebinqin/U-2-Net),metric[Saliency-Evaluation-Toolbox](https://github.com/jiwei0921/Saliency-Evaluation-Toolbox)

## Running the Evaluation

To evaluate the model on different datasets, you should download our [pretrained model](https://drive.google.com/drive/folders/13yOAB7wB1-W3ialY4MVZXS_SbwNAAY4A?usp=drive_link) and write the location of checkpoint in "pretrain_model_G" of options/test/dataset_name.yml.

```
python test.py -opt ./options/test/TURBID.yml
```
## Running the training
```
python train.py -opt ./options/test/TURBID.yml
```



