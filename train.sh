export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data_root ./ln_data/ --batch_size 8 --lr 0.0001 --output_dir ./outputs/2021_08_12/referit --lr_visu_tra 0.00001 --lr_visu_cnn 0.00001 --lr_bert 0.00001 --model_name TransVG --dataset referit --max_query_len 40 --dec_layer 0 --backbone resnet50 --detr_model ./checkpoints/detr-r50.pth --aug_crop --aug_scale --aug_translate
