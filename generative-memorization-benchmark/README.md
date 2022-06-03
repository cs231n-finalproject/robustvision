# Generative Memorization Benchmark
This repo contains the performance evaluation code used in the [Kaggle Generative Dog Images Competition](https://www.kaggle.com/c/generative-dog-images/overview), the first-ever large-scale generative modeling competition.
The goal of the competition is to demonstrate the efficacy of the Memorizaion-informed Fréchet Inception Distance (MiFID) for detecting intentional memorization in a competition setting.
The competition submissions are published as a dataset with manual labels of the memorization technique applied.
It is openly released at [Generative Dog Images](https://www.kaggle.com/andrewcybai/generative-dog-images).

## Requirements
You can install the recommended environment as follows:
```
conda env create -f env.yml -n genmem
```
The pretrained model weights and embedded training/testing features needed to be combined as follows:
```
cat ./data/models_misc.tar.gz.parta* > ./data/models_misc.tar.gz
tar -xvf ./data/models_misc.tar.gz -C ./data
```

## Quick start
To evaluate the competition stats of a generated set of images (PNG)
```
python src/competition_scoring.py [path_to_image_dir]
```
Special thanks to Tensorflow for the pretrained classification model weights [link](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

## Citation
This repo is established for the following research project. 
Please cite our work if you use it.
```
@inproceedings{bai2021genmem,
  author = {Ching-Yuan Bai and Hsuan-Tien Lin and Colin Raffel and Wendy Chih-wen Kan},
  title = {On Training Sample Memorization: Lessons from Benchmarking Generative Modeling with a Large-scale Competition},
  booktitle = {Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)},
  year = 2021,
  month = aug,
  keyword = {deep, kdd}
}
```
