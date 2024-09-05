rm -r swin_base_patch4_window7_224

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \--cfg configs/swin/swin_base_patch4_window7_224.yaml --train-path /mnt/sdb1/ariel/Desktop/Disco/part1 --val-path /mnt/sdb1/ariel/Desktop/Disco/part1 --normalization-path /mnt/sdb1/ariel/Desktop/Scripts-Pre/normalization_values.csv --accumulation-steps 1
