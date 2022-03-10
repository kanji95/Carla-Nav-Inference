import os
episodes = list(range(15,25))
checkpoint = './saved_model/baseline_cross_attention_convlstm_class_level_combo_07_Mar_16-07.pth'
backbone = 'convlstm'
maps = ['Town03', 'Town03', 'Town03', 'Town03', 'Town01', 'Town05', 'Town03', 'Town10HD', 'Town05', 'Town05', 'Town10HD', 'Town03',
        'Town03', 'Town10HD', 'Town03', 'Town10HD', 'Town02', 'Town07', 'Town03', 'Town01', 'Town10HD', 'Town10HD', 'Town01', 'Town10HD', 'Town10HD']

for i in range(len(episodes)):
    print(f"python inference_model.py --img_backbone {backbone} --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target network \
            --sampling 5 --num_frames 8 --stop_criteria confidence --confidence 150 --map {maps[episodes[i]]} --num_preds 3 --spawn {episodes[i]} --command")
    os.system(f"python inference_model.py --img_backbone {backbone} --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target network \
            --sampling 5 --num_frames 8 --stop_criteria confidence --confidence 150 --map {maps[episodes[i]]} --num_preds 3 --spawn {episodes[i]} --command")
