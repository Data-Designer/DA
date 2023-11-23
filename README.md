<h1 align="center"> Bi-discriminator Domain Adversarial Neural Networks with Class-Level Gradient Alignment </h1>

## About Our Work

Update: 2023/11/23: We have created a repository for the paper titled *Bi-discriminator Domain Adversarial Neural
Networks with Class-Level Gradient Alignment*, which has been submitted to the **IEEE Transactions on Systems, Man, and Cybernetics: Systems (SMCA) **. In this repository, we offer the original sample datasets, preprocessing scripts, and algorithm files to showcase the reproducibility of our work.

![framework_page-0001](https://s2.loli.net/2023/11/23/7ydEiXmPaAKtfnu.jpg)

## Requirements

- Python == 3.8.10
- Pytorch == 1.2.0
- timm== 0.9.11
- scikit-learn== 1.2.2
- wilds== 2.0.0

## Data Sets

The structure of the data set should be like

```
data
|_ clef
|  |_ b
|  |_ c
|  |_ i
|  |_ p
|  |_ list
|_ digit
|_ |_ ...
|_ visda
|_ |_ ...
|_ cifa
|_ |_ ...
```

Due to the copyright limitations, we have not uploaded the data.  You can seek permission from the organizer according to the link given or download it directly from their website.

[ImageClef Dataset](https://www.imageclef.org/datasets)

[CIFAR Dataset](https://cs.stanford.edu/~acoates/stl10/)

[Visda2017 Dataset](https://ai.bu.edu/visda-2017/)

[Digits Dataset](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md)

## RUN

You should update the log and data reading directories in the configuration file initially. 

```powershell
# unzip all files into the DA directory
# run BACG
python main.py
# run Fast-BACG
python main_mem.py
```

## Contributors ✨

Many thanks to the data preprocessing pipeline in the following published papers.

[Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library/tree/master)； [Deep Evidential Learning](https://github.com/aamini/evidential-deep-learning); [CDGM](https://github.com/lijin118/CGDM)