import os
episodes = list(range(0, 34))

checkpoint = './saved_model/new_clip/clip_ViT-B_32_class_level_combo_multi_head_hd_512_sf_10-1_tf_20_22_Jun_11_17.pth'
maps = ['Town05', 'Town03', 'Town10HD', 'Town01', 'Town05', 'Town03',
        'Town02', 'Town03', 'Town05', 'Town02', 'Town05', 'Town05',
        'Town05', 'Town01', 'Town01', 'Town10HD', 'Town02', 'Town05',
        'Town05', 'Town03', 'Town07', 'Town03', 'Town05', 'Town05',
        'Town01', 'Town02', 'Town01', 'Town10HD', 'Town02', 'Town01',
        'Town01', 'Town10HD', 'Town02', 'Town05']

command = True
for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 1 --traj_frames 10 --attn_type multi_head --one_in_n 1\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 3 --spawn {episodes[i]} {'--command' if command else ''}")
