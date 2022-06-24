import os
episodes = [19,27]+list(range(34,34))

checkpoint = './saved_model/nomap_clip/clip_ViT-B_32_class_level_combo_multi_head_hd_512_sf_10-1_tf_20_22_Jun_11_14.pth'
maps = ['Town05', 'Town03', 'Town10HD', 'Town01', 'Town05', 'Town03',
        'Town02', 'Town03', 'Town05', 'Town02', 'Town05', 'Town05',
        'Town05', 'Town01', 'Town01', 'Town10HD', 'Town02', 'Town05',
        'Town05', 'Town03', 'Town07', 'Town03', 'Town05', 'Town05',
        'Town01', 'Town02', 'Town01', 'Town10HD', 'Town02', 'Town01',
        'Town01', 'Town10HD', 'Town02', 'Town05']


command = True
for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target distance \
            --num_frames 1 --traj_frames 20 --attn_type multi_head --one_in_n 1 --min_distance 1.0 --considered_distance 30\
            --sampling 2 --sampling_type constant --stop_criteria consistent --min_confidence 0.4 --infer_dataset test --distance 7 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}")
