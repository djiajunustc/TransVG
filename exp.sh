export CUDA_VISIBLE_DEVICES=0,1,2

# # RefCOCOg g-split
python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone vit_small --detr_model ../DETR-Models/detr_vit_s_depth_12_v1_no_enc.pth --bert_enc_num 12 --detr_enc_num 0 --hidden_dim 384 --dim_feedforward 1536 --nheads 6 --dataset gref --max_query_len 40 --output_dir outputs/2022_02_15/refcocog_gsplit_vit_s_depth12_v1_no_enc --imsize 640 --epochs 90 --lr_drop 60


