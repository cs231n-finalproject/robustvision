from mmdet.apis import init_detector, inference_detector, show_result_pyplot

config_file = 'conformer/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'conformer/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
img = 'conformer/mmdetection/demo/demo.jpg'
# inference the demo image
result = inference_detector(model, img)
show_result_pyplot(model, img, result)

print('successful!')