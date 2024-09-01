# Introduction

Basic implementation of Controller RNN from [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) and [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012).

- The project is based on the idea of applying Neural Architecture Search on Transformer model to find the optimal hyperparameter that would achieve best accuracy. 
- Neural Architecture Search is a technique to automate the architectural design process of a Neural Network.
- Transformer model is a neural network known for its self-attention mechanism that allows the model to capture relationships between distant elements in a sequence

# Model Setup
**NAS:**
A Neural Architecture Search uses Reinforcement Learning to learn the optimal hyperparameters to train a child network 

<img src="https://github.com/peterzrz/NAS_Transformer/blob/master/images/img1.png">

- We plan to utilize the transformer model, specifically the Vision Transformer (ViT) model, as a replacement of CNN model used in the previous NAS paper.

- To be specific, we would generate hyperparameters for the trans- former and test if the NAS method is still robust in this circumstance

- The main task environment we will have our model interact with is the CIFAR-10 data-set, which contains 60, 000 32x32 color images split up into 10 classes of 6000 images. 

- Our NAS will search for a model that fits CIFAR-10, measure the performance of the best transformer model to visual problems, and compare it to the experiments done with a standard CNN.

# Methodology
Our investigation will be to explore the performance of the Neural Architecture Search (NAS) with a Transformer on different types of datasets ranging from visual to text-based. 

- We will first build the RNN controllers as described by the paper. This RNN model will then predict the optimal hyperparameters, the # of heads, for a transformer model on the Cifar-10 dataset and the Portuguese to English dataset. 
- Then, we will evaluate the transformer model based on its accuracy, and utilize this metric as a feedback to adjust the RNN prediction. 
- Eventually, we fine-tune our best transformer model among all trials and examine its validation accuracy. 

Our idea is to compare the results from the transformer and the results from the paper’s CNN model, and discuss in detail how the complication of transformers might have given rise to the differences.

# Implementation Details
**Data Preprocessing:**
- Image normalization, Resizing, Random flipping/cropping
**Divide each image into 144 patches**
**Encoder**
- Encode the patches with positional embedding
**Four Multi-Attention Layers**
**Hyperparameters to tune:**
- num_heads: # attention heads in each layer [2, 4, 6, 8]
- dff: The feed forward projection dimension [32, 64, 128]


# Usage
At a high level : For full training details, please see `train.py`.
```python
# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='kernel', values=[1, 3])
state_space.add_state(name='filters', values=[16, 32, 64])

# create the managers
controller = Controller(tf_session, num_layers, state_space)
manager = NetworkManager(dataset, epochs=max_epochs, batchsize=batchsize)

# For number of trials
  sample_state = ...
  actions = controller.get_actions(sample_state)
  reward = manager.get_reward(actions)
  controller.train()
```

# Result

**Experiment 1 : CNN vs. Transformers**
| ------------  | Training Loss | Test Loss    |
| ------------- | ------------- | ------------- | 
| CNN | 0.742 | 0.662 |
| Transformers |0.713| 0.650 |

<img src="https://github.com/peterzrz/NAS_Transformer/blob/master/images/img2.png>

**Experiment 2: Varying # CNN layers**
| #Layers  | 4 | 5  | 6 |  7 |
| ------------- | ------------- | ------------- |------ | ------ |
| Iterations to Converge | 50 | 56 | 63 | 79|

<img src="https://github.com/peterzrz/NAS_Transformer/blob/master/images/img3.png>

**Experiment 3: NAS on CIFAR 100**
| Dataset  | CIFAR10 | CIFAR100  |
| ------------- | ------------- | ------------- |
| Iterations to Converge | 50 | 43 |

<img src="https://github.com/peterzrz/NAS_Transformer/blob/master/images/img4.png>


# Conclusion
- The lack of significant improvement in accuracy when using the more advanced ViT model shows the inability of NAS to work with ViT model. The Controller Loss graph also backs this claim up. However, this could also because of our insufficient training iterations.

- We observed that the NAS approach, when being applied to CNN, is very robust to the increase of hyperparameters to tune. The convergence time and #hyperparameters tend to have a linear relationship. In addition, it is insensitive to the dataset’s complexity.

- NAS has some major limitations. There is no obvious method to tune its “own” hyperparameters: exploration constant, # training epochs, etc. It is also sensitive to the random hyperparameter initialization.

- As a next step, we want to devote more GPU on Transformer to account for its abnormal behaviors and discover methods of finding optimal constants/hyperparameters for the Controller.


# Acknowledgements
Code heavily inspired by [wallarm/nascell-automl](https://github.com/wallarm/nascell-automl)
