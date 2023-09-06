# Environment Setup 

```bash
conda create --name weakclip python=3.7 -y
conda activate weakclip
```

First please install `torch==1.8.1` `torchvision==0.9.1` follow the pytorch official install documents according to your cuda version.

```bash
# cuda11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Then install `mmcv-full==1.3.17` follow the mmcv official install documents according to the torch version and your cuda version

```bash
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
```

Finally install other requirements:

```bash
pip install -r requirements.txt

# install pydencecrf
pip install --force-reinstall cython==0.29.36
pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git

# get deeplab retrain code
git clone https://github.com/Yingyue-L/deeplabv1-resnet38
ln -s ../data deeplabv1-resnet38
```

