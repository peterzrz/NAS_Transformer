import numpy as np
import keras
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class CustomSchedule(tf.compat.v1.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)

class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, epochs=20, child_batchsize=128, acc_beta=0.8, clip_rewards=0.0):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0

    def get_rewards(self, model_fn, actions, policy_sess_test):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.
            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given actions
        '''
        # existing_graph = tf.compat.v1.get_default_graph()

        # with tf.compat.v1.Session(graph=existing_graph).as_default() as network_sess:
        # with tf.compat.v1.Session(graph=tf.compat.v1.Graph()).as_default() as network_sess:
        K.set_session(policy_sess_test)

        # generate a submodel given predicted actions
        model = model_fn(actions)  # type: Model
        x_train, y_train, x_val, y_val = self.dataset
        
        learning_rate = 0.001
        weight_decay = 0.0001
        
        batch_size = 256
        num_epochs = 10 
        #optimizer = keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)

        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )
        
        checkpoint_filepath = "/tmp/checkpoint.weights.h5"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )


        # evaluate the model
        loss, acc = model.evaluate(x_val, y_val, batch_size=self.batchsize)

        # compute the reward
        reward = (acc - self.moving_acc)

        # if rewards are clipped, clip them in the range -0.05 to 0.05
        if self.clip_rewards:
            reward = np.clip(reward, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if self.beta > 0.0 and self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * acc
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0

            reward = np.clip(reward, -0.1, 0.1)

        print()
        print("Manager: EWA Accuracy = ", self.moving_acc)

        # clean up resources and GPU memory
        # network_sess.close()

        return reward, acc