export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7



python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 47770 \
eval.py \
--batch_size 16 \
--vit_model tiny \
--bert_model bert-base-uncased \
--max_query_len 40 \
--dataset gref_umd \
--eval_set test \
--reg_out_type reg_token \
--language_modulation cross_attn \
--without_visual_mask \
--eval_model outputs/gref_umd_transvg_plusplus_tiny_cross_attn_lr_1e-4/best_checkpoint.pth \
--output_dir outputs/gref_umd_transvg_plusplus_tiny_cross_attn_lr_1e-4