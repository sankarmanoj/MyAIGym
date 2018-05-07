import gym
import tensorflow as tf
import random
import numpy as np
random.seed()
env = gym.make('CartPole-v1')
all_times= []
for i_episode in range(20000):
    observation = env.reset()
    observations = []
    actions = []
    for t in range(100):
        # env.render()
        action = random.randint(0,1)
        observations.append(observation)
        observation, reward, done, info = env.step(action)
        t_action = [0,0]
        t_action[action] = 1
        actions.append(t_action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            if t+1 > 80:
                all_times.append((t+1,observations,actions))
            break
all_times.sort(key = lambda x : x[0],reverse = True)
training_data = all_times
print len(training_data)

tf_input = tf.placeholder(dtype = tf.float32,shape = [None,4])
tf_actual = tf.placeholder(dtype = tf.float32,shape = [None,2])

l1 = tf.layers.dense(inputs = tf_input, units = 5 , activation = tf.tanh)
l2 = tf.layers.dense(inputs = l1, units = 3 , activation = tf.tanh)
output = tf.layers.dense(inputs = l2, units = 2 , activation = tf.tanh)



loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=tf_actual))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()

print "Start Learning"
with tf.Session() as sess:
    sess.run(init)
    for step in range(100):
        for data in training_data:
            sess.run(train_op,feed_dict = { tf_input : data[1] , tf_actual : data[2]})
    print "Done Learning"
    for i_episode in range(20):
        observation = env.reset()
        t=0
        while True:
            t+=1
            env.render()
            observation, reward, done, info = env.step(action)
            action = np.argmax(sess.run(output,feed_dict={tf_input:[observation]}))
            observations.append(observation)
            actions.append(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                all_times.append((t+1,observations,actions))
                raw_input("Enter")
                break
