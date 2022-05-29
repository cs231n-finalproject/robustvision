# Robustvision

In the past 2 years, self-supervised pre-training for vision representation [1] [10] has outperformed supervised pretraining [11] in computer vision tasks such as image classification, inspired by transformer-based pre-trained models in NLP. Transformer-based models used in NLP have shown promising results in computer vision benchmarks in fields such as Object Detection [3], Video Classification [4], Image Classification [5] and Image Generation [6]. Transformer-based methods showed the power of learning global representation of the images. However, CNN models still have the advantage of learning local feature representation which is important for instance segmentation tasks. There are some recent works combining CNN and Transformer (a.k.a., Conformer) which could potentially lead to a robust vision model leveraging the best of both worlds [2].
This methodology is similar to the Two-Streams Hypothesis in Neuroscience, which claims that the brain has a dorsal and ventral pathway [12] [13] which separately process spatial, location-based information and recognition / identification respectively. The corollaries to current deep learning approaches would be transformer-based models for the recognition, classification tasks, and CNN-based models for spatial locality. Following this path, the goal of this project is to find a single architecture combining self-supervised / supervised pre-trained vision models and variations of the Conformer architecture to Object Detection, Instance Segmentation and Style Transformation, and conclude a solution of one robust vision model that works across different tasks.

<img width="913" alt="image" src="https://user-images.githubusercontent.com/28990806/170882276-5c4da835-2748-48b8-b4a7-8cfea30f9a2a.png">



# Convergence of fused two towers:

## Metrics:
<img width="285" alt="image" src="https://user-images.githubusercontent.com/28990806/170881527-3cd7440d-faa9-4a8c-a3d6-a3584d0653cb.png">

## Feature Map:

<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881570-fa12ed82-f0b2-423f-8d30-dd49983b2f42.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881571-ef52aca2-50d7-469a-b0c7-b461cdc97383.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881585-27a0b86c-cbbd-46d0-82e7-5da364ce8d04.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881601-d56b34d9-8db3-4a01-b775-cd52da162e52.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881613-0385df88-cfed-4fc8-9d73-507b30bae9c1.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881637-14b37aed-18b8-4540-b0e5-4d96e3311092.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881654-1b2ca2c9-f10a-4101-85a1-12150f0667b3.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881667-21973c15-c9f2-48e5-af7f-82c2bb4b0bd9.png">
<img width="407" alt="image" src="https://user-images.githubusercontent.com/28990806/170881683-dc76d80b-6b86-4b2d-ae18-24093da999a4.png">
<img width="403" alt="image" src="https://user-images.githubusercontent.com/28990806/170881710-0335d7ed-c7c3-460e-8f5e-79e5e997300a.png">
<img width="403" alt="image" src="https://user-images.githubusercontent.com/28990806/170881722-dc9f1bc7-2654-4f1e-ba94-95242e6b0d7b.png">
<img width="403" alt="image" src="https://user-images.githubusercontent.com/28990806/170881738-57062d1a-3a71-4251-a596-6313ea8dc9a6.png">
## Weight:
