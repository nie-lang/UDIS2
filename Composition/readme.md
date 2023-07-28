## Train on UDIS-D
Before training, the warped images and corresponding masks should be generated in the warp stage.

Then, set the training dataset path in Composition/Codes/train.py.

```
python train_H.py
```

## Test on UDIS-D
The pre-trained model of warp is available at [Google Drive](https://drive.google.com/file/d/1OaG0ayEwRPhKVV_OwQwvwHDFHC26iv30/view?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1qCGegzvxtzri6GiG7mNw6g)(Extraction code: 1234).

Set the testing dataset path in Composition/Codes/test.py.

```
python test.py
```
The composition masks and final fusion results on UDIS-D will be generated and saved at the current path.


## Test on other datasets
Set the 'path/' in Composition/Codes/test_other.py. 
```
python test_other.py
```
The results will be generated and saved at 'path'.
