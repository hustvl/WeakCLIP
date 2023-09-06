# Training
## VOC12

Download the [PMM Pseudo Mask](https://drive.google.com/file/d/1fbxB17SC4oPb8MBxoThCBvEmBWk8ZoSC/view?usp=share_link) and [MCTformer Pseudo Mask](https://drive.google.com/file/d/1VqfWU9CAEeA4tEElmANW6hzoHCWlUyqO/view?usp=share_link).

```bash
bash tools/dist_train.sh configs/voc_weakclip_vit-b_512x512_20k_mct.py 2
``` 

## COCO14

Download the [PMM Pseudo Mask](https://drive.google.com/file/d/1L8pzwcNOmilK7xAe7fPgHS0eH8sbS8-W/view?usp=share_link) and [MCTformer Pseudo Mask](https://drive.google.com/file/d/1G5nTedBLvjBQw4FULg-f-z7ob-GluuWj/view?usp=share_link). 

```bash
bash tools/dist_train.sh configs/coco_weakclip_vit-b_512x512_40k_mct.py 4
``` 


# Pseudo Mask Generation

## VOC12

```bash
bash tools/dist_test.sh \
configs/voc_weakclip_vit-b_512x512_20k_mct.py \
work_dirs/voc_weakclip_vit-b_512x512_20k_mct/train_74p03_iter_8000.pth 1 --eval mIoU --aug-test
```

```bash
python tools/make_crf.py \
--list trainaug.txt \
--data-path data \
--predict-dir prob_npy \
--predict-png-dir pred_png \
--num-cls 21 \
--dataset voc12
```


## COCO

```bash
cd segmentation
bash tools/dist_test.sh \
configs/coco_weakclip_vit-b_512x512_40k_mct.py \
work_dirs/coco_weakclip_vit-b_512x512_40k_mct/46p07_iter_29000.pth 1 --eval mIoU --aug-test
```

```bash
python make_crf.py \
--list train.txt \
--data-path data \
--predict-dir prob_npy \
--predict-png-dir pred_png \
--num-cls 91 \
--dataset coco
```