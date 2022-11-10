#!/bin/bash

export data_path='ogb2022-dataset/Pretrained_3D_ViSNet_dataset'
export rdkit_data_path='ogb2022-dataset/Pretrained_3D_ViSNet_dataset/rdkit_data/'
export tc=True

bash OGB_ViSNet/preprocess/preprocess.sh

for i in 0 1
do
CUDA_VISIBLE_DEVICES=0 python train.py --conf examples/OGB-Pretrained-3D-ViSNet.yaml --load-model checkpoints/Pretrained_3D_ViSNet_ckpt/Pretrained_3D_ViSNet_ckpt_$i.ckpt --dataset-root ogb2022-dataset/Pretrained_3D_ViSNet_dataset/ --log-dir results/test-challenge-pt-visnet-$i/ --inference-dataset test-challenge --is-submit --task inference
done

for i in {0..19}
do
CUDA_VISIBLE_DEVICES=0 python train.py --conf examples/OGB-Transformer-M-ViSNet.yaml --load-model checkpoints/Transformer_M_ViSNet_ckpt/Transformer_M_ViSNet_ckpt_$i.ckpt --dataset-root ogb2022-dataset/Transformer_M_ViSNet_dataset/ --log-dir results/test-challenge-tm-visnet-$i/ --inference-dataset test-challenge --is-submit --task inference
done

python ensemble.py results/ $data_path/pcqm4m-v2/raw/data.csv.gz $data_path/pcqm4m-v2/split_dict.pt

# clean tmp files
rm -r *.tmp
rm -r *.xyz