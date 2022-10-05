# Knowledge Graph Embedding by Relational Rotation and Complex Convolution for Link Prediction

This source code for ConvRot is based on OpenKE - An Open-source Framework for Knowledge Embedding implemented with PyTorch.

More information is available on OpenKE website
[http://openke.thunlp.org/](http://openke.thunlp.org/)

If you use the framework, please cite the following [paper](http://aclweb.org/anthology/D18-2024):

```
 @inproceedings{han2018openke,
   title={OpenKE: An Open Toolkit for Knowledge Embedding},
   author={Han, Xu and Cao, Shulin and Lv Xin and Lin, Yankai and Liu, Zhiyuan and Sun, Maosong and Li, Juanzi},
   booktitle={Proceedings of EMNLP},
   year={2018}
 }
```

The base code of OpenKE is mainly contributed (in chronological order) by [Xu Han](https://github.com/THUCSTHanxu13), [Yankai Lin](https://github.com/Mrlyk423), [Ruobing Xie](http://nlp.csai.tsinghua.edu.cn/~xrb/), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/), [Xin Lv](https://github.com/davidlvxin), [Shulin Cao](https://github.com/ShulinCao), [Weize Chen](https://github.com/chenweize1998), [Jingqin Yang](https://github.com/yjqqqaq).

The new version of OpenKE is mainly contributed by [Nam Le](https://github.com/nhutnamhcmus)

## Overview about OpenKE

This is an Efficient implementation based on PyTorch for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE composes 4 repositories:

OpenKE-PyTorch: the project based on PyTorch, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0"> OpenKE-Tensorflow1.0</a>: OpenKE implemented with TensorFlow, also providing the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD.

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE.


*** **UPDATE** ***

We are now developing a new version of OpenKE-PyTorch. The project has been completely reconstructed and is faster, more extendable and the codes are easier to read and use now. If you need get to the old version, please refer to branch [OpenKE-PyTorch(old)](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch(old)).

*** **New Features** ***

- RotatE
- More enhancing strategies (e.g., adversarial training)
- More scripts of the typical models for the benchmark datasets.
- More extendable interfaces

## Overview ConvRot

ConvRot model to address these shortcomings by integrating a 2D convolution. Specifically, we perform convolution on embeddings of entities and relations to obtain support vector embeddings. These vectors are then combined into an element-wise rotation from the head entity to the tail entity using the Hadamard product, allowing the model to capture local interactions between entities and relations through the neural network while still ensuring intuitiveness through a roto-transformation in the link prediction. In addition, we offer two different strategies to design the complex convolution module and show their effect on the model performance. This proposed method is evaluated on standard benchmark datasets and achieves significantly improved results on MRR and Hits@K (K= 1, 3, 10). Overall, the link prediction performance of our model improves approximately by 5–7%. Besides, the ConvRot model is also considered separately on many relation types, such as one-to-one, one-to-many, many-to-one, and many-to-many.

## Models

OpenKE (Tensorflow):

*	RESCAL, HolE
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD

OpenKE (PyTorch):

*	 RESCAL
*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD
*  SimplE (Fixed follwing: [https://github.com/thunlp/OpenKE/issues/151](https://github.com/thunlp/OpenKE/issues/151))
*	 RotatE
*  NConRot (Our proposed model)
*  HConvRot (Our proposed model)

We welcome any issues and requests for model implementation and bug fix.

## Experimental Settings

For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. The scores of those corrupted triplets are first computed by the models and then sorted by the order. Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). The mean rank (MRR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.

Because some corrupted triplets may be in the training set and validation set. In this case, those corrupted triplets may be ranked above the test triplet, but this should not be counted as an error because both triplets are true. Hence, we remove those corrupted triplets appearing in the training, validation or test set, which ensures the corrupted triplets are not in the dataset. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. The mean rank (MRR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.

More details of the above-mentioned settings can be found from the papers [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. Hence, we also provide the experimental setting named "[type constraint](https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf)" to corrupt entities with some limited entity sets determining by their relations.

## Experiments

We have provided the hyper-parameters of some models to achieve the state-of-the-art performace (Hits@10 (filter)) on FB15K237 and WN18RR. These scripts can be founded in the file train_convrot_FB15K237_adv.py and train_convrot_WN18RR_adv.py. Up to now, these models include TransE, TransH, TransR, TransD, DistMult, ComplEx.

<strong> We provide a new version of OpenKE-Pytorch new</strong>: That allow user can valid their own model on validation set of each benchmark dataset.

<strong> We are still trying more hyper-parameters and more training strategies (e.g., adversarial training and label smoothing regularization) for these models. </strong> Hence, this table is still in change. We welcome everyone to help us update this table and hyper-parameters.


## Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/) for latest PyTorch Version

2. Clone this project

```
git clone https://github.com/lnthanhhcmus/ConvRot
```

3. Compile C++ files: Requirement g++/gcc for latest version

```bash
bash make.sh
```
4. Quick Start

For FB15k-237, FB15k and YAGO3-10:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python train_convrot_FB15K237_adv.py
```

For WN18RR and WN18:

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python train_convrot_WN18RR_adv.py
```

## Data

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .
  **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

* For testing, datasets contain additional two files (totally five files):

  test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(e1, e2, rel)*** .

  valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(e1, e2, rel)*** .

  type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K

## Authors

Thanh Le, Nam Le, Bac Le (Faculty of Information Technology, University of Science, Ho Chi Minh City, Vietnam and Vietnam National University, Ho Chi Minh City, Vietnam)

## CRediT authorship contribution statement

Thanh Le: Conceptualization, Methodology, Writing- Original Draft, Validation, Supervision. Nam Le: Methodology, Software, Writing- Original Draft, Formal analysis, Investigation, Visualization. Bac Le: Supervision.

## Declaration of competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

