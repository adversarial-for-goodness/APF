## APF

This is a tensorflow implementation of paper "[Adversarial Privacy-preserving Filter](https://arxiv.org/abs/2007.12861)" at  *ACM Multimedia 2020*.



## Copyright

This code is intended only for personal privacy protection or academic research. 

## Running Environment

- Python 3.7.7 
- pillow, scipy, numpy ...
- tensorflow 1.15.0
- mxnet 1.3.1 (only needed when reading mxrec file)

## Preparation

### Data Prepare

The official InsightFace (ArcFace) project share their training data and testing data in the [DataZoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo). This data is in mxrec format, you can transform it to tfrecord format with [./data/generateTFRecord.py](https://github.com/adversarial-for-goodness/APF/blob/master/data/generateTFRecord.py) by the following script:

```
python generateTFRecord.py 
--mode=mxrec
--image_size=112
--read_dir=$DIRECTORY_TO_THE_TRAINING_DATA$
--save_path=$DIRECTORY_TO_SAVE_TFRECORD_FILE$/xxx.tfrecord
```

Then you can run [./data/createdata_zx.py](https://github.com/adversarial-for-goodness/APF/blob/master/data/createdata_zx.py) to create training data for APF, which includes 100,000 face images of 2,000 subjects (you can generate more training data if you have bigger memory):

```
python createdata_zx.py
--subjects=2000
--read_path=$DIRECTORY_TO_SAVE_TFRECORD_FILE$/xxx.tfrecord
--save_path=$PATH_TO_SAVE_FINAL_TRAINING_FILE$
```

### Model Prepare

The proposed APF includes probe model and server model. In this implementation, we use [MobileFaceNet](https://github.com/sirius-ai/MobileFaceNet_TF) as probe model and [InsightFace](https://github.com/luckycallor/InsightFace-tensorflow) as server model. You can download pre-trained model weights of [MobileFaceNet](https://github.com/sirius-ai/MobileFaceNet_TF/tree/master/arch/pretrained_model/) and [InsightFace](https://pan.baidu.com/s/1v1L3c7cEs_GyqPYH9WhNKA) to your model directory. 

## Pretrained Model

Here we open our pretrained models for easier application ([Baidu](https://pan.baidu.com/s/1oZAGxkOuMa5aQElxrdNtgQ) password:txy1).
It acheives better performance than paper claimed in LFW dataset.

You can evaluate a pretrained model with [main.py](https://github.com/adversarial-for-goodness/APF/blob/master/main.py), for example:

```
python main.py 
--mode='test' 
--test_insightface_model_path=$PATH_TO_PRETRAINED_INSIGHTFACE_MODEL$
--test_mobilefacenet_model_path=$PATH_TO_PRETRAINED_MOBILEFACENET_MODEL$
--test_model_path=$PATH_TO_PRETRAINED_APF_MODEL$
--test_data=$DIRECTORY_OF_EMORE$/lfw.bin
```


## Train Your Own Model

The following script starts training:

```
python main.py 
--mode='train' 
--insightface_model_path=$PATH_TO_PRETRAINED_INSIGHTFACE_MODEL$
--mobilefacenet_model_path=$PATH_TO_PRETRAINED_MOBILEFACENET_MODEL$
--train_data=$PATH_TO_SAVE_FINAL_TRAINING_FILE$
--val_data=$DIRECTORY_OF_EMORE$/lfw.bin
--train_model_ouput=$PATH_TO_SAVE_APF_MODEL$
```


## Citation
```
@inproceedings{zhang2020adversarial,
  title={Adversarial Privacy-preserving Filter},
  author={Zhang, Jiaming and Sang, Jitao and Zhao, Xian and Huang, Xiaowen and Sun, Yanfeng and Hu, Yongli},
  booktitle="Proceedings of the 28th ACM International Conference on Multimedia",
  year={2020}
}
```

## References

1. [https://github.com/deepinsight/insightface]
2. [https://github.com/luckycallor/InsightFace-tensorflow]
3. [https://github.com/sirius-ai/MobileFaceNet_TF]
