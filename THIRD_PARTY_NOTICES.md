# Third-party notices

## MAE ViT-B/16 encoder

`foveawam-ontimebench.pt` includes the encoder from Meta's
`mae_pretrain_vit_base.pth`, released with *Masked Autoencoders Are Scalable
Vision Learners* by Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr
Dollár and Ross Girshick. The upstream project credits Meta Platforms, Inc. and
affiliates. FoveaWAM uses the frozen encoder at BF16 precision and stores its
weights as FP32 tensors in the deployment bundle. The encoder retains its
CC BY-NC 4.0 license. See the [original checkpoint](https://github.com/facebookresearch/mae#fine-tuning-with-pre-trained-checkpoints)
and the [license text](https://huggingface.co/JiaqiYin/FoveaWAM/blob/main/MAE-LICENSE.txt).

## DINOv3 training supervision

Meta's DINOv3 ViT-L/16 supplied future patch features as auxiliary targets
during training. The deployment bundle omits the teacher parameters. Section
1(b) of Meta's agreement requires acknowledgment in research publications and
a copy of the agreement when redistributing DINO Materials or their derivatives.
See the [teacher checkpoint](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
and the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md).
