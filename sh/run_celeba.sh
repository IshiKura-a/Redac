CUDA_VISIBLE_DEVICES=0 python main.py \
  --root_dir /data/home/tangzihao/dataset/CelebA \
  --k 4 \
  --e 0.999 \
  --feature layer4 \
  --log_dir /data/home/tangzihao/model/group_dro_resnet/ \
  --lr 0.0001 \
  --weight_decay=0.08
