import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
import DQN
import random

env = gym.make('CartPole-v0')
env._max_episode_steps = 10001
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
#minimum distance from target and prediction
dis = 0.9
#def standardize()

def get_copy_var_ops(*, src_scope_name="main", dest_scope_name="target"):
    #get variable copying operations(as tensorflow node)
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    return op_holder

def replay_train(pred_net, target_net, train_batch):
    #get next_state(Q) from target and update the graph.
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, done in train_batch:
        #Q is [[4]] shaped array.
        Q = pred_net.predict(state)

        if done: Q[0, action] = reward
        else: Q[0, action] = reward + dis * np.max(target_net.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return pred_net.update(x_stack, y_stack)

def botPlay(pred_net):
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = np.argmax(pred_net.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print(f"Total score: {reward_sum}") 
            break
        
def main():
    bufferMemory = 50000
    maxEpisode = 500

    actionBuffer = deque()

    with tf.Session() as sess:
        pred_net = DQN.DQN(input_size, output_size, sess, "main")
        target_net = DQN.DQN(input_size, output_size, sess, "target")
        sess.run(tf.global_variables_initializer())

        copy_ops = get_copy_var_ops(src_scope_name="main", dest_scope_name="target")
        sess.run(copy_ops)

        for episode in range(maxEpisode):
            #state is [1, 4] shaped array.
            state = env.reset()
            e = 1. / ((episode / 10) + 1)
            step_count = 0
            done = False

            while not done:
                #take action
                if np.random.rand(1) < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(pred_net.predict(state))
                #get next state
                next_state, reward, done, _ = env.step(action)
                if done: reward = -100
                #store action
                actionBuffer.append((state, action, reward, next_state, done))
                if len(actionBuffer) > bufferMemory:
                    actionBuffer.popleft()
                
                state = next_state
                step_count += 1
                if step_count > 10000: break

            print("Episode: {} steps: {}".format(episode, step_count))
            if step_count > 10000: pass

            if episode % 10 == 1:
                for _ in range(50):
                    minibatch = random.sample(actionBuffer, 10)
                    loss, _ = replay_train(pred_net, target_net, minibatch)

                print(f'Loss: {loss}')
                sess.run(copy_ops)
        
        botPlay(pred_net)

if __name__ == "__main__":
    main()