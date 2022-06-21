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

```
 python .\inference_model.py --img_backbone vit_small_patch16_224 --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005  --checkpoint .\saved_model\baseline_vit_small_patch16_224_0.27560.pth --glove_path E:\carla\carla\CARLA_0.9.12\glove\glove\ --target network --sampling 5 --stop_criteria confidence --confidence 150 --map Town10HD
```


```
python .\inference_model.py --img_backbone timesformer --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005  --checkpoint .\saved_model\baseline_timesformer_24_Feb_02-15.pth --glove_path E:\carla\carla\CARLA_0.9.12\glove\glove\ --target network --sampling 10 --stop_criteria confidence --confidence 10 --num_frames 8 
```

```
python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 6 --traj_frames 10 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}
```

```
checkpoint = './saved_model/conv3d_baseline_class_level_combo_multi_head_hd_384_sf_10_tf_20_05_Apr_09_00.pth'
command = True
for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone conv3d_baseline --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask_dual \
            --num_frames 4 --traj_frames 10 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}")
```

Network mode requires very low confidence and very low threshold.
Mask mode requires a good amount of confidence (about 50 for vit, 100-150 for deeplab). Threshold of about 0.001 for vit, 0.2 for deeplab.

Both of the above require low threshold as the thresholding is to only divide the segmentation map in components.


Trajectory requires high threshold. Confidence doesn't matter.
