export CUDA_VISIBLE_DEVICES=0,1,2,3

# # RefCOCOg g-split
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --batch_size 16 --bert_model bert-base-uncased --vit_model tiny --visual_model_stride 32 --lr_bert 0.00001 --dataset gref --aug_scale --aug_crop --aug_translate --visual_pretrained ../DETR-Models/detr_vit_tiny_for_transvg_640_no_window.pth --vl_hidden_dim 192 --vl_depth 3 --max_query_len 40 --output_dir outputs/2022_03_28/refcocog_gsplit_vit_tiny_cross_lang_depth3_no_rt --imsize 640 --epochs 90 --lr_drop 60 --without_reg_token --separate_qkv


# python train.py --batch_size 16 --bert_model bert-base-uncased --vit_model tiny --visual_model_stride 32 --lr_bert 0.00001 --dataset gref --aug_scale --aug_crop --aug_translate --visual_pretrained ../DETR-Models/detr_vit_tiny_for_transvg_640_no_window.pth --vl_hidden_dim 192 --vl_depth 3 --max_query_len 40 --output_dir outputs/2022_03_28/refcocog_gsplit_vit_tiny_cross_lang_depth3_no_rt --imsize 640 --epochs 90 --lr_drop 60 --without_reg_token --separate_qkv

# # RefCOCOg umd-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone vit_small --detr_model ../DETR-Models/detr_vit_small_depth12_v4_enc_0_gref.pth --bert_enc_num 12 --detr_enc_num 0 --hidden_dim 384 --dim_feedforward 1536 --nheads 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/2022_02_26/refcocog_usplit_vit_s_depth12_v4_no_enc --imsize 640 --epochs 90 --lr_drop 60
