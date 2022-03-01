import os
episodes = list(range(25))
checkpoint = './saved_model/baseline_avg_concat_vit_small_patch16_224_bce_1.00000.pth'
maps = ['Town03', 'Town03', 'Town03', 'Town03', 'Town01', 'Town05', 'Town03', 'Town10HD', 'Town05', 'Town05', 'Town10HD', 'Town03',
        'Town03', 'Town10HD', 'Town03', 'Town10HD', 'Town01', 'Town07', 'Town03', 'Town01', 'Town10HD', 'Town10HD', 'Town01', 'Town10HD', 'Town10HD']

for i in range(len(episodes)):
    os.system(f"python inference_model.py --img_backbone vit_small_patch16_224 --hidden_dim 384 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path E:/carla/carla/CARLA_0.9.12/glove/glove/ --target network \
            --sampling 5 --stop_criteria confidence --confidence 150 --map {maps[episodes[i]]} --num_preds 3 --spawn {episodes[i]} --command")
