export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

output_dir=./outputs/test_referit

python -m torch.distributed.launch --nproc_per_node=8 --use_env evaluation.py --data_root ./ln_data/ --batch_size 32 --output_dir $output_dir --model_name TransVG --dataset referit --max_query_len 20 --dec_layer 0 --backbone resnet101 --num_workers 4 --imsize 640 --eval_set test --eval_model /data/dengjj/VG/iccv_exp/TransVG/outputs/20210315/referit/r101_transvg/checkpoint.pth 
