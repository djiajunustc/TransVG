export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# ReferItGame
python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ../released_models/TransVG_referit.pth --output_dir ./outputs/referit_r50


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testA --eval_model ../released_models/TransVG_unc.pth --output_dir ./outputs/refcoco_r50


# # RefCOCO+
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testA --eval_model ../released_models/TransVG_unc+.pth --output_dir ./outputs/refcoco_plus_r50


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ../released_models/TransVG_gref.pth --output_dir ./outputs/refcocog_gsplit_r50


# # RefCOCOg u-split
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ../released_models/TransVG_gref_umd.pth --output_dir ./outputs/refcocog_usplit_r50