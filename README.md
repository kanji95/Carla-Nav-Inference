## Hidden layer sizes:

```
vit_tiny_patch16_224: 192
vit_small_patch16_224: 384
deeplabv3_resnet50: 2048
dino_resnet50: 256
```

## Inference Example:

```
python .\inference_model.py --glove_path E:\carla\carla\CARLA_0.9.12\glove\glove\ --checkpoint .\saved_model\baseline_deeplabv3_resnet50_18_Feb_06-11.pth --img_backbone deeplabv3_resnet50 --hidden_dim 2048 --map Town10HD --threshold 0.1 --sync
```