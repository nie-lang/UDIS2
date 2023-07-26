# <p align="center">Parallax-Tolerant Unsupervised Deep Image Stitching ([paper](https://arxiv.org/abs/2302.08207))</p>
<p align="center">Lang Nie*, Chunyu Lin*, Kang Liao*, Shuaicheng Liu`, Yao Zhao*</p>
<p align="center">* Institute of Information Science, Beijing Jiaotong University</p>
<p align="center">` School of Information and Communication Engineering, University of Electronic Science and Technology of China</p>

![image](https://github.com/nie-lang/UDIS2/blob/main/fig1.png)

## Dataset (UDIS-D)
We use the UDIS-D dataset to train and evaluate our method. Please refer to [UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) for more details about this dataset.


## Code
#### Requirement
* numpy 1.19.5
* pytorch 1.7.1
* scikit-image 0.15.0
* tensorboard 2.9.0

We implement this work with Ubuntu, 3090Ti, and CUDA11. Refer to [environment.yml](https://github.com/nie-lang/UDIS2/blob/main/environment.yml) for more details.

#### How to run it
Similar to UDIS, we also implement this solution in two stages:
* Stage 1 (unsupervised warp): please refer to  [Warp/readme.md]([https://github.com/nie-lang/UnsupervisedDeepImageStitching/blob/main/ImageAlignment/ImageAlignment.md](https://github.com/nie-lang/UDIS2/blob/main/Warp/readme.md)).
* Stage 2 (unsupervised composition): please refer to .



## Meta
If you have any questions about this project, please feel free to drop me an email.

NIE Lang -- nielang@bjtu.edu.cn
```
@article{nie2023learning,
  title={Learning Thin-Plate Spline Motion and Seamless Composition for Parallax-Tolerant Unsupervised Deep Image Stitching},
  author={Nie, Lang and Lin, Chunyu and Liao, Kang and Liu, Shuaicheng and Zhao, Yao},
  journal={arXiv preprint arXiv:2302.08207},
  year={2023}
}
```

## References
[1] L. Nie, C. Lin, K. Liao, M. Liu, and Y. Zhao, “A view-free image stitching network based on global homography,” Journal of Visual Communication and Image Representation, p. 102950, 2020.  
[2] L. Nie, C. Lin, K. Liao, and Y. Zhao. Learning edge-preserved image stitching from multi-scale deep homography[J]. Neurocomputing, 2022, 491: 533-543.   
[3] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Unsupervised deep image stitching: Reconstructing stitched features to images[J]. IEEE Transactions on Image Processing, 2021, 30: 6184-6197.   
[4] L. Nie, C. Lin, K. Liao, S. Liu, and Y. Zhao. Deep rectangling for image stitching: a learning baseline[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022: 5740-5748.   
