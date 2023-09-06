# Dataset Prepare

## Pre-trained CLIP Models

Download the pre-trained CLIP models ([VIT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)) and save them to the `pretrained` folder.

## Pascal VOC 2012
- First download the Pascal VOC 2012 datasets use the scripts in the `data` dir.

```bash
cd data
sh download_and_convert_voc12.sh
```
- Then download SBD annotations from [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip).

The folder structure is assumed to be:
```bash
WeakCLIP
├── data
│   ├── download_and_convert_voc12.sh
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── SegmentationClassAug
│   │   │   ├── SegmentationClassAugPseudoMaskMCT
```

## COCO 2014 
- First download the COCO 2014 datasets use the scripts in the `data` dir.

```bash
cd data
sh download_and_convert_coco.sh
cp val_5000.txt COCO14/voc_format
```
The folder structure is assumed to be:
```bash
WeakCLIP
├── data
│   ├── download_and_convert_coco.sh
│   ├── VOCdevkit
│   ├── COCO14
│   │   ├── images
│   │   ├── voc_format
│   │   │   ├── class_labels
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   │   ├── val_5000.txt
│   │   │   ├── cocoPGTMCT
```