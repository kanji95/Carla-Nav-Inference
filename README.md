# Ground then Navigate: Language-guided Navigation in Dynamic Scenes (ICRA 2023)

Official Code | [Paper](https://arxiv.org/pdf/2209.11972.pdf) | [Video](https://youtu.be/bSwtb6APGns)

## Abstract
> We investigate the Vision-and-Language Navigation (VLN) problem in the context of autonomous driving in outdoor settings. We solve the problem by explicitly grounding the navigable regions corresponding to the textual command. At each timestamp, the model predicts a segmentation mask corresponding to the intermediate or the final navigable region. Our work contrasts with existing efforts in VLN, which pose this task as a node selection problem, given a discrete connected graph corresponding to the environment. We do not assume the availability of such a discretised map. Our work moves towards continuity in action space, provides interpretability through visual feedback and allows VLN on commands requiring finer manoeuvres like "park between the two cars". Furthermore, we propose a novel meta-dataset CARLA-NAV to allow efficient training and validation. The dataset comprises pre-recorded training sequences and a live environment for validation and testing. We provide extensive qualitative and quantitive empirical results to validate the efficacy of the proposed approach.

![Screenshot 2023-06-20 at 3 06 06 PM](https://github.com/kanji95/Carla-Nav/assets/30688360/3866fa1d-bd8c-47b4-89cc-13fb9966e4d4)

## Dataset 

Coming Soon

## Implementation Details

### Hidden layer sizes:

```
vit_tiny_patch16_224: 192
vit_small_patch16_224: 384
deeplabv3_resnet50: 2048
dino_resnet50: 256
```

### Inference Example:

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
# clip context
python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 6 --traj_frames 10 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}
```

```
# clip no context
checkpoint = './saved_model/nomap_clip/clip_ViT-B_32_class_level_combo_multi_head_hd_512_sf_10-6_tf_20_0.20166.pth'
python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 6 --traj_frames 10 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 50 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}
```

```
# clip single with context
checkpoint = './saved_model/new_clip/clip_ViT-B_32_class_level_combo_multi_head_hd_512_sf_10-1_tf_20_22_Jun_11_17.pth'
python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 1 --traj_frames 10 --attn_type multi_head --one_in_n 1\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset val --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 3 --spawn {episodes[i]} {'--command' if command else ''}


```

```
checkpoint = './saved_model/conv3d_baseline_class_level_combo_multi_head_hd_384_sf_10_tf_20_05_Apr_09_00.pth'
for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone conv3d_baseline --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask_dual \
            --num_frames 4 --traj_frames 10 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}")
```

```
checkpoint = './saved_model/iros_class_level_combo_multi_head_hd_384_sf_10-4_tf_20_20_Jun_20_26.pth'
python inference_model.py --img_backbone iros --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 1 --traj_frames 20 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}
```

```
checkpoint = './saved_model/rnrcon_class_level_combo_multi_head_hd_384_sf_10-4_tf_20_21_Jun_21_34.pth'
python inference_model.py --img_backbone rnrcon --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 1 --traj_frames 20 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset val --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 3 --spawn {episodes[i]} {'--command' if command else ''}
```

Network mode requires very low confidence and very low threshold.
Mask mode requires a good amount of confidence (about 50 for vit, 100-150 for deeplab). Threshold of about 0.001 for vit, 0.2 for deeplab.

Both of the above require low threshold as the thresholding is to only divide the segmentation map in components.


Trajectory requires high threshold. Confidence doesn't matter.
