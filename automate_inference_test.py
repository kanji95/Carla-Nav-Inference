import os
episodes = list(range(0, 25))
checkpoint = './saved_model/conv3d_baseline_class_level_combo_multi_head_hd_384_sf_10_tf_20_05_Apr_09_00.pth'
maps = ['Town05', 'Town03', 'Town10HD', 'Town01', 'Town05', 'Town03',
       'Town02', 'Town03', 'Town05', 'Town02', 'Town05', 'Town05',
       'Town05', 'Town01', 'Town01', 'Town10HD', 'Town02', 'Town05',
       'Town05', 'Town03', 'Town07', 'Town02', 'Town05', 'Town05',
       'Town01']
       
command = True
for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone conv3d_baseline --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target mask \
            --num_frames 4 --traj_frames 10 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 5 --spawn {episodes[i]} {'--command' if command else ''}")
