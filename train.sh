export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# -------------------------  RefCOCOg u-split --------------------------

# language-adapter
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
train.py \
--batch_size 4 \
--vit_model tiny \
--bert_model bert-base-uncased \
--lr 0.0001 \
--lr_vision 0.00001 \
--lr_language 0.00001 \
--dataset gref_umd \
--aug_scale --aug_crop --aug_translate \
--visual_pretrained ../ViTDet-Models/vitdet_tiny_for_lvit_640_global_12.pth \
--max_query_len 40 \
--epochs 60 --lr_drop 45 \
--separate_qkv \
--reg_out_type reg_token \
--language_modulation cross_attn \
--without_visual_mask \
--output_dir outputs/gref_umd_transvg_plusplus_tiny_cross_attn_lr_1e-4