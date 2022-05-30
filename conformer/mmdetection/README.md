## Notice
The code is forked from official [project](https://github.com/open-mmlab/mmdetection). **So the basic install and usage of mmdetection can be found in** [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md). 

pip3 install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html
pip3 install -r requirements/build.txt
pip3 install -v -e .
pip3 install instaboostfast
pip3 install git+https://github.com/cocodataset/panopticapi.git
pip3 install git+https://github.com/lvis-dataset/lvis-api.git
pip3 install -r requirements/optional.txt
mkdir data
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

We just add Conformer as a backbone in `mmdet/models/backbones/Conformer.py`.

At present, we use the feature maps of different stages in the CNN branch as the input of FPN, so that it can be quickly applied to the detection algorithm based on the feature pyramid. **At the same time, we think that how to use the features of Transformer branch for detection is also an interesting problem.**

## Training and inference under different detction algorithms
We provide some config files in `configs/`. And anyone can use Conformer to replace the backbone in the existing detection algorithms. We take the `Faster R-CNN` algorithm as an example to illustrate how to perform training and inference:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=1
GPU_NUM=6

CONFIG='~/robustvision/conformer/mmdetection/configs/faster_rcnn/faster_rcnn_conformer_small_patch32_fpn_1x_coco.py'
WORK_DIR='~/robustvision/conformer/mmdetection/work_dir/faster_rcnn_conformer_small_patch32_lr_1e_4_fpn_1x_coco_1344_800'

# Train
python3 -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./conformer/mmdetection/tools/train.py --config ${CONFIG} --work-dir ${WORK_DIR} --gpus ${GPU_NUM}  --launcher pytorch --cfg-options model.pretrained='~/robustvision/conformer/mmdetection/pretrain_models/Conformer_small_patch32.pth' model.backbone.patch_size=32

# Test on multiple cards
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=50040 --use_env ./tools/test.py ${CONFIG} ${WORK_DIR}/latest.pth --launcher pytorch  --eval bbox

# Test on single card
#./tools/test.py ${CONFIG} ${WORK_DIR}/latest.pth --eval bbox
```

Here, we use the `Conformer_small_patch32` as backbone network, whose pretrain model weight can be downloaded from [baidu (k7q5)](https://pan.baidu.com/s/1pum_kOOwQYn404ZeGzjMlg) or [google drive](https://drive.google.com/file/d/1UrvRg2hnXsie_z_y39Xavdts4qfrwZ1E/view?usp=sharing). And the results are shown as following:

| Method        | Parameters | MACs   | FPS | Bbox mAP | Model link | Log link |
| ------------ | ---------- | ------ | ------ | --------- | ---- |---- |
| Faster R-CNN | 55.4 M     | 288.4 G | 13.5 | 43.1    | [baidu](https://pan.baidu.com/s/1lkZy_FTLeCRg3rVH8dOKOA)(7ax9) [google](https://drive.google.com/drive/folders/1gCvcW3Zhqq8KK5GnAr9So7-5uJwnrZcA?usp=sharing) | [baidu](https://pan.baidu.com/s/10HTtS8FozMSYfHJv8L2H5w)(ymv4)|
| Mask R-CNN | 58.1 M     | 341.4 G | 10.9 | 43.6   | [baidu](https://pan.baidu.com/s/1wqvhbq4ePAPIZFqE0aCWEQ)(qkwq) [google](https://drive.google.com/drive/folders/1mjoReWPoBSMUIjBQE5VlhQf0XZ2sE7J-?usp=sharing)|[baidu](https://pan.baidu.com/s/1lSq7hMTSA8fN7WNXTZqp7g)(gh2v)|
|PAA (1x single scale)| - | - | - | 46.5 | (coming soon) | -|
|Cascade Mask RCNN (1x single scale)| - | - | - | 47.3 | (coming soon) | -|

## Update Detection Performance

| Method        | Schedule | Parameters | MACs   | FPS | Bbox mAP | Segm mAP |
| ------------ | ----- | ----- | ------ | ------ | --------- | ---- |
Faster R-CNN | 1x | 55.4 M |   288.4 G | 13.5 | 43.7 | - |
Faster R-CNN | 3x | 55.4 M |   288.4 G | 13.5 | 46.1 | - |
