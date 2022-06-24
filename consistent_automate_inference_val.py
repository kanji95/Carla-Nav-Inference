import os
episodes = [0, 6, 7, 11, 15, 20, 22]+list(range(25, 25))

checkpoint = './saved_model/new_clip/clip_ViT-B_32_class_level_combo_multi_head_hd_512_sf_10-4_tf_20_11_Jun_14_40_0.23184.pth'
maps = ['Town05', 'Town05', 'Town01', 'Town02', 'Town05', 'Town05',
        'Town03', 'Town10HD', 'Town02', 'Town05', 'Town05', 'Town10HD',
        'Town01', 'Town10HD', 'Town01', 'Town03', 'Town10HD', 'Town07',
        'Town03', 'Town01', 'Town05', 'Town03', 'Town10HD', 'Town02',
        'Town05']


command = True
for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 4 --traj_frames 20 --attn_type multi_head --one_in_n 10\
            --sampling 5 --sampling_type constant --stop_criteria consistent --confidence 150 --min_confidence 25 --infer_dataset val --distance 4 --map {maps[episodes[i]]}\
                 --num_preds 3 --spawn {episodes[i]} {'--command' if command else ''}")
