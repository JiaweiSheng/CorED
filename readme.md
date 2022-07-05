# CorED
Source code for SIGIR2022 paper: [*CorED: Incorporating Type-level and Instance-level Correlations for Fine-grained Event Detection*](https://doi.org/10.1145/3477495.3531956).

Event detection (ED) is a pivotal task for information retrieval, which aims at identifying event triggers and classifying them into pre-defined event types.
This paper simultaneously incorporates both the type-level and instance-level event correlations, and proposes a novel framework, termed as CorED.
Specifically, we devise an adaptive graph-based type encoder to capture type-level correlations, learning type representations not only from their training data but also from their relevant types, thus leading to more informative type representations especially for the low-resource types.
Besides, we devise an instance interactive decoder to capture instance-level correlations, which predicts event instance types conditioned on the contextual typed event instances, leveraging co-occurrence events as remarkable evidence in prediction.
Empirical results demonstrate the unity of both type-level and instance-level correlations, and the model achieves effectiveness performance on both benchmarks.


# Requirements

We conduct our experiments on the following environments:

```
python 3.6
CUDA: 9.0
GPU: Tesla T4
pytorch == 1.1.0
transformers == 4.9.1
```

# Datasets

We adopt MAVEN and ACE-2005 as our datasets.
The original MAVEN dataset can be accessed at [this repo](https://github.com/THU-KEG/MAVEN-dataset).

# How to run

To run the code, you could run as following steps:

1. Put pretrained language models into ``./plm/bert-base-uncased``. Put original MAVEN dataset into ``./dataset/maven/maven``.

2. Run data preprocess as follows:

```
cd ./dataset/maven
python data_process.py
```

3. Conduct training/validation/testing as follows:

```
CUDA_VISIBLE_DEVICES=0 python -u main_cls.py --data_type maven --prefix exp_model  --do_train true --do_valid true --do_test true 
```

The hyper-parameters are recorded in ``./utils/params.py``. 


# Citation

If you find this code useful, please cite our work:

```
@inproceedings{Sheng2022:CorEE,
  title     = {CorED: Incorporating Type-level and Instance-level Correlations for Fine-grained Event Detection},
  author    = {Jiawei Sheng and Rui Sun and Shu Guo and Shiyao Cui and Jiangxia Cao and Lihong Wang and Tingwen Liu and Hongbo Xu},
  booktitle = {SIGIR},
  year      = {2022}
}
```

