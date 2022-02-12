export CUDA_VISIBLE_DEVICES=0,1,2

# # RefCOCOg g-split
python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone vit_small --detr_model ../DETR-Models/detr_vit_s_depth_12_enc_3.pth --bert_enc_num 12 --detr_enc_num 3 --hidden_dim 384 --dim_feedforward 1536 --nheads 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_vit_s --imsize 896 --lr_bert 0


