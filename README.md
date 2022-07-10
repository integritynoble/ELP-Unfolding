# Ensemble Learning Priors Driven Deep Unfolding for Scalable Snapshot Compressive Imaging [PyTorch]
## Abstract 
Snapshot compressive imaging (SCI) can record the 3D information by a 2D measurement and from this 2D measurement to  reconstruct the original 3D information by reconstruction algorithm. As we can see, the reconstruction algorithm plays a vital role in SCI. Recently, deep learning algorithm show its outstanding ability, outperforming the traditional algorithm. Therefore, to improve deep learning algorithm reconstruction accuracy is an inevitable topic for SCI. Besides, deep learning algorithms are usually limited by scalability, and a well trained model in general can not be applied to new systems if lacking the new training process. To address these problems, we develop the ensemble learning priors to further improve the reconstruction accuracy and propose the scalable learning to empower  deep learning the scalability just like the traditional algorithm. What's more, our algorithm has achieved the state-of-the-art results, outperforming existing algorithms. Extensive results on both simulation and real datasets demonstrate the superiority of our proposed algorithm.
# Comparison of some results
![](./video/scalable/1024_gif/Beauty_1024×1024.gif)
![](./video/scalable/1024_gif/ShakeNDry_1024×1024.gif)
![](./video/scalable/1024_gif/ReadySetGo_1024×1024.gif)
![](./video/scalable/1024_gif/YachtRide_1024×1024.gif)

# Principle of ELP-Unfolding
![principle](fig/principle.jpg)
![principle](fig/single.jpg)
![principle](fig/ensemble.jpg)
![principle](fig/prior.jpg)


## Prerequisite

```shell
$ pip install pytorch=1.9
$ pip install tqdm
$ pip install random
$ pip install wandb
$ pip install argparse
$ pip install scipy
```
## Test
### For the Benchmark

Download our trained model from the [Google Drive](https://drive.google.com/file/d/1yxX_jjBQF73LfFFMV_2GlkpUNosWq_C2/view?usp=sharing) and place it under the log_dir (your path) folder. Then you should modify （init and pres) channel number 64 into the 512, which is the original number the paper. 512 can help you get the better result as those in paper. 64 can help you run in a GPU with low memory.
```shell
cd ./ELP_Unfolding
python test.py  or  bash test.sh
```

### For the Scalable

Download our trained model from the [Google Drive](https://drive.google.com/file/d/1--fcQrfeVKJnQzFLWCIQ6o0HrdCq_Cmk/view?usp=sharing) and place it under the log_dir (your path)folder. Then you should modify （init and pres) channel number 64 into the 512, which is the original number the paper. 512 can help you get the better result as those in paper. 64 can help you run in a GPU with low memory.
```shell
cd ./ELP_Unfolding/scalable
python test.py  or  bash test.sh
```

## Train
Download our trained model from the [Google Drive](https://drive.google.com/drive/folders/1nFI5LFqgowvlBUdO8O1jypq2xqTu2EcD?usp=sharing) and place it under the traindata folder. 
### For the Benchmark
```shell
cd ./ELP_Unfolding
python test.py  or  bash test.sh
```

### For the Scalable

```shell
cd ./ELP_Unfolding/scalable
python test.py  or  bash test.sh
```

## Results
### For Benchmark dataset

![Results](fig/Benchmark.jpg)

### For scalable dataset

![Results](fig/512.jpg)
![Results](fig/1024.jpg)

## Citation
If you find the code helpful in your resarch or work, please cite the following paper.
```
@inproceedings{,
  title={},
  author={},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```