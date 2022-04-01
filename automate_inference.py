import os
episodes = list(range(21, 22))
backbone = 'convattn'
checkpoint = './saved_model/convattn_class_level_combo_dot_product_hd_192_sf_10_tf_20_31_Mar_07_48.pth'
maps = ['Town03', 'Town03', 'Town03', 'Town03', 'Town01', 'Town05', 'Town03', 'Town10HD', 'Town05', 'Town05', 'Town10HD', 'Town03',
        'Town03', 'Town10HD', 'Town03', 'Town10HD', 'Town02', 'Town07', 'Town03', 'Town01', 'Town10HD', 'Town10HD', 'Town01', 'Town10HD', 'Town10HD']

for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone {backbone} --hidden_dim 192 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target network \
            --num_frames 4 --one_in_n 10 --traj_frames 20 --attn_type dot_product \
            --sampling 5 --stop_criteria distance --confidence 320 --distance 25 --map {maps[episodes[i]]} --num_preds 5 --spawn {episodes[i]} --command")
