export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone vit_tiny --detr_model ../DETR-Models/detr_vit_tiny_for_transvg_epoch0074.pth --bert_enc_num 12 --detr_enc_num 0 --hidden_dim 192 --dim_feedforward 768 --nheads 6 --dataset gref --max_query_len 40 --output_dir outputs/2022_03_04/refcocog_gsplit_vit_tiny --imsize 640 --epochs 90 --lr_drop 60 --bert_model bert-base-uncased

python train.py --batch_size 8 --bert_model bert-base-uncased --vit_model vit_tiny --visual_model_stride 16 --lr_bert 0.00001 --dataset gref --aug_scale --aug_crop --aug_translate --visual_pretrained ../DETR-Models/detr_vit_tiny_for_transvg_embed_40_40.pth --vl_hidden_dim 192 --vl_dim_feedforward 768 --vl_nheads 3 --vl_enc_layers 6 --max_query_len 40 --output_dir outputs/2022_03_04/refcocog_gsplit_vit_tiny --imsize 640 --epochs 90 --lr_drop 60 

# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone vit_small --detr_model ../DETR-Models/detr_vit_small_depth12_v4_enc_0_gref.pth --bert_enc_num 12 --detr_enc_num 0 --hidden_dim 384 --dim_feedforward 1536 --nheads 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/2022_02_26/refcocog_usplit_vit_s_depth12_v4_no_enc --imsize 640 --epochs 90 --lr_drop 60
