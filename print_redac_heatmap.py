import argparse
import os
import faulthandler


import torchvision
from torchvision.models import ResNet50_Weights

from modules.dataset import CelebADataset
from utils.logger import visualize_redac_cluster


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, default="resnet-50")
    parser.add_argument("--root_dir", type=str, default=f'/data/home/tangzihao/dataset/CelebA')
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--e", type=float, default=0.999)
    parser.add_argument("--feature", type=str, default="layer1")
    parser.add_argument("--log_dir", type=str, default=f'/data/home/tangzihao/model/group_dro_resnet')

    args = parser.parse_args()
    log_path = f'{args.log_dir}/k{args.k}_e{args.e}_{args.feature}.png'

    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    full_dataset = CelebADataset(
        root_dir=args.root_dir,
        target_name="Blond_Hair",
        confounder_names=["Male"],
        augment_data=False,
        backbone=model,
        k=args.k,
        epsilon=args.e,
        feature=args.feature,
        apply_cluster=True
    )

    visualize_redac_cluster(full_dataset.attrs_df, full_dataset.attr_names, full_dataset.cluster_array, args.k,
                            log_path)


if __name__ == "__main__":
    faulthandler.enable()
    main()
