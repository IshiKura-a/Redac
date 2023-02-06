CUDA_VISIBLE_DEVICES=3 python main.py \
  --dataset WaterBird \
  --root_dir /data/home/tangzihao/dataset/waterbird_complete95_forest2water2 \
  --k 4 \
  --e 0.9996 \
  --feature layer4 \
  --log_dir /data/home/tangzihao/model/redac/wbs \
  --weight_decay 0.1 \
  --n_epochs 300 \
  --lr 1e-5