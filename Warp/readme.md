## Train on UDIS-D
Set the training dataset path in Warp/Codes/train.py.

```
python train.py
```

## Test on UDIS-D
The pre-trained model of warp is available at [Google Drive](https://drive.google.com/file/d/1GBwB0y3tUUsOYHErSqxDxoC_Om3BJUEt/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1Fx6YnQi9B2wvP_TOVAaBEA)(Extraction code: 1234).
#### Calculate PSNR/SSIM
Set the testing dataset path in Warp/Codes/test.py.

```
python test.py
```

#### Generate the warped images and corresponding masks
Set the training/testing dataset path in Warp/Codes/test_output.py.

```
python test_output.py
```
The warped images and masks will be generated and saved at the original training/testing dataset path. The results of average fusion will be saved at the current path.

## Test on other datasets
When testing on other datasets with different scenes and resolutions, we apply the iterative warp adaption to get better alignment performance.

Set the 'path/img1_name/img2_name' in Warp/Codes/test_other.py. (By default, both img1 and img2 are placed under 'path')
```
python test_other.py
```
The results before/after adaption will be generated and saved at 'path'.

