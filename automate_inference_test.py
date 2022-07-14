import os
episodes = list(range(0, 1))

checkpoint = './saved_model/clip_ViT-B_32_class_level_combo_multi_head_hd_512_sf_10-2_tf_20_09_Jul_12_38.pth'
maps = ['Town05', 'Town03', 'Town10HD', 'Town01', 'Town05', 'Town03',
        'Town02', 'Town03', 'Town05', 'Town02', 'Town05', 'Town05',
        'Town05', 'Town01', 'Town01', 'Town10HD', 'Town02', 'Town05',
        'Town05', 'Town03', 'Town07', 'Town03', 'Town05', 'Town05',
        'Town01', 'Town02', 'Town01', 'Town10HD', 'Town02', 'Town01',
        'Town01', 'Town10HD', 'Town02', 'Town05']

command = True
for i in range(len(episodes)):
    os.system(f"CUDA_VISIBLE_DEVICES=2 python inference_model.py --img_backbone clip_ViT-B/32 --hidden_dim 512 --image_dim 224 --mask_dim 224 --traj_dim 224 --sync --threshold 0.00005 \
        --checkpoint {checkpoint} --glove_path ~/glove/glove/ --target mask \
            --num_frames 2 --traj_frames 20 --attn_type multi_head --one_in_n 10\
            --sampling 5 --stop_criteria confidence --confidence 150 --min_confidence 25 --infer_dataset test --distance 5 --map {maps[episodes[i]]}\
                 --num_preds 3 --spawn {episodes[i]} {'--command' if command else ''}")
