import numpy as np
import csv

import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import to_categorical

import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text

from controller import Controller, StateSpace
from manager import NetworkManager
from model import model_fn
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

tf.compat.v1.disable_eager_execution()

# create a shared session between Keras and Tensorflow
policy_sess = tf.compat.v1.Session()
K.set_session(policy_sess)
print(policy_sess)

NUM_LAYERS = 4  # number of layers of the state space
MAX_TRIALS = 250  # maximum number of models generated

MAX_EPOCHS = 10  # maximum number of epochs to train
CHILD_BATCHSIZE = 128  # batchsize of the child models
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training

MAX_TOKENS=128
BUFFER_SIZE = 20000
BATCH_SIZE = 64

  
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels


def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

# construct a state space
state_space = StateSpace()

# add states
state_space.add_state(name='activation', values=["relu", "softmax"]) # add the activation tuning

# print the state space being searched
state_space.print_state_space()

# prepare the training data for the NetworkManager
#fetch data
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
#Set up tokenizer
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

previous_acc = 0.0
total_reward = 0.0

with policy_sess.as_default():
    print(policy_sess)
    # create the Controller and build the internal policy network
    controller = Controller(policy_sess, NUM_LAYERS, state_space,
                            reg_param=REGULARIZATION,
                            exploration=EXPLORATION,
                            controller_cells=CONTROLLER_CELLS,
                            embedding_dim=EMBEDDING_DIM,
                            restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(train_batches, val_batches, epochs=MAX_EPOCHS, child_batchsize=CHILD_BATCHSIZE, clip_rewards=CLIP_REWARDS,
                         acc_beta=ACCURACY_BETA)

# get an initial random state space if controller needs to predict an
# action from the initial state
state = state_space.get_random_state_space(NUM_LAYERS)
print("Initial Random State : ", state_space.parse_state_space_list(state))
print()

# clear the previous files
controller.remove_files()

# train for number of trails
for trial in range(MAX_TRIALS):
    with policy_sess.as_default():
        K.set_session(policy_sess)
        actions = controller.get_action(state)  # get an action for the previous state

    # print the action probabilities
    state_space.print_actions(actions)
    print("Predicted actions : ", state_space.parse_state_space_list(actions))

    # build a model, train and get reward and accuracy from the network manager
    print(policy_sess)
    reward, previous_acc = manager.get_rewards(model_fn, state_space.parse_state_space_list(actions))
    print("Rewards : ", reward, "Accuracy : ", previous_acc)

    with policy_sess.as_default():
        K.set_session(policy_sess)

        total_reward += reward
        print("Total reward : ", total_reward)

        # actions and states are equivalent, save the state and reward
        state = actions
        controller.store_rollout(state, reward)

        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step()
        print("Trial %d: Controller loss : %0.6f" % (trial + 1, loss))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+') as f:
            data = [previous_acc, reward]
            data.extend(state_space.parse_state_space_list(state))
            writer = csv.writer(f)
            writer.writerow(data)
    print()

print("Total Reward : ", total_reward)