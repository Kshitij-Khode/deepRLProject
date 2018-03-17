#!/usr/bin/env python
from keras.models       import Sequential
from keras.layers.core  import Dense, Dropout, Activation
from keras              import optimizers
from keras.layers       import Lambda
from keras.models       import Model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization        import BatchNormalization

import tensorflow    as tf
import numpy         as np
import keras.backend as K

import gym, sys, copy, argparse, collections, time, random, os

class QNetwork():

    def __init__(self):



        self.lrate  = 1e-4
        self.dlrate = 1e-5

        self.model = Sequential()

        # Definition of the architecture of different kind of Q networks requested
        if self.qnmodel == 'dqn':
            self.model.add(Dense(30, input_shape=(self.state_dim,), init='lecun_uniform'))
            self.model.add(LeakyReLU(alpha=0.01))
            self.model.add(Dense(30, init='lecun_uniform'))
            self.model.add(LeakyReLU(alpha=0.01))
            self.model.add(Dense(30, init='lecun_uniform'))
            self.model.add(LeakyReLU(alpha=0.01))
            self.model.add(Dense(self.num_act, init='lecun_uniform'))

        if self.qnmodel == 'linear':
            self.model.add(Dense(256, input_shape=(self.state_dim,), init='lecun_uniform'))
            self.model.add(Activation("linear"))
            self.model.add(Dense(self.num_act, init='lecun_uniform'))

        if self.qnmodel == 'ddqn':
            self.model.add(Dense(30, input_shape=(self.state_dim,), init='lecun_uniform'))
            self.model.add(LeakyReLU(alpha=0.01))
            self.model.add(Dense(30, init='lecun_uniform'))
            self.model.add(LeakyReLU(alpha=0.01))
            self.model.add(Dense(30, init='lecun_uniform'))
            self.model.add(LeakyReLU(alpha=0.01))
            self.model.add(Dense(self.num_act, init='lecun_uniform'))

            # Instantiate the exact Q network as used in DQN part except replace the penultimate layer
            # with 2 layers, 1 for value stream other for advantage.
            layer      = self.model.layers[-2]
            na         = self.model.output._keras_shape[-1]
            y          = Dense(na + 1, activation='linear')(layer.output)

            # Combine outputs of the 2 streams according the equation 9 in \cite{wang2015dueling}
            op_layer   = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(self.num_act,))(y)
            self.model = Model(inputs=self.model.input, outputs=op_layer)

        self.optimizer = optimizers.Adam(lr=self.lrate)
        self.model.compile(loss="mse", optimizer=self.optimizer)
        self.model.summary()

    def initSession():
        num_threads = os.environ.get('OMP_NUM_THREADS')

        if num_threads:
            config = tf.ConfigProto(intra_op_parallelism_threads=num_threads)
            config.gpu_options.allow_growth=True
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True

        return tf.Session(config=config)

    def createModel():
        inputL = tf.placeholder(tf.float32, shape=(inShape[0], inShape[1], window), 'inputL')

        with tf.variable_scope('lay1'):         hidden = tf.layers.dense(inputL, 128, activation=tf.nn.relu)

    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
    # Make output layers
    with tf.variable_scope("layer3"):
        logits = tf.layers.dense(hidden, 2)
    # Take the action with the highest activation
    with tf.variable_scope("output"):
        action = tf.argmax(input=logits, axis=1)

    return state_ph, action, logits

        with tf.name_scope(model_name):
            model = Sequential()
            model.add(Conv2D(filters = 32, kernel_size = 8, strides = (4,4), input_shape=(input_shape[0], input_shape[1], window), name='conv1'))
            model.add(Activation('relu', name='relu1'))
            model.add(Conv2D(filters = 64, kernel_size = 4, strides = (2,2), name='conv2'))
            model.add(Activation('relu', name='relu2'))
            model.add(Conv2D(filters = 64, kernel_size = 3, strides = (1,1), name='conv3'))
            model.add(Activation('relu', name='relu3'))
            model.add(Flatten())
            model.add(Dense(512, name='dense1'))
            model.add(Activation('relu', name='relu4'))
            model.add(Dense(num_actions, name='dense2'))
            model.add(Activation('linear', name='linear1'))

    elif (network == 'linear'):
        with tf.name_scope(model_name):
            model = Sequential()
            model.add(Reshape((input_shape[0]*input_shape[1]*window, ), input_shape=(input_shape[0], input_shape[1], window), name='reshape1'))
            model.add(Dense(num_actions, name='dense1'))
            model.add(Activation('linear', name='linear1'))

    print(model.summary())

    return model

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights.
        self.model.save_weights('./model/%s_weights.ckpt' % suffix, overwrite=True)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        pass

    def load_model_weights(self, weight_file):
        # Helper funciton to load model weights.
        self.model.load_weights(weight_file)

    def get_qvals(self, state):
        if self.env_name == 'MountainCar-v0': return self.model.predict(np.array([[state[0], state[1]]]))[0]
        if self.env_name == 'CartPole-v0': return self.model.predict(np.array([[state[0], state[1], state[2], state[3]]]))[0]

    def update_net(self, states, q_lbls):
        loss = self.model.train_on_batch(np.array(states), np.array(q_lbls))
        return loss

    def print_qmap(self):
        if self.env_name == 'MountainCar-v0':
            print('--- Q Map Start ---')
            for p in np.linspace(self.min_position, self.max_position, self.pos_int):
                for v in np.linspace(0, self.max_speed, self.vel_int):
                    print('Pos: %.5s, Vel: %.5s, Qs: %s' % (p, v, self.get_qvals([p,v])))
            print('--- Q Map End ---')

        if self.env_name == 'CartPole-v0':
            print('--- Q Map Start ---')
            for p in np.linspace(self.min_pos, self.max_pos, self.pos_int):
                for t in np.linspace(self.min_theta, self.max_theta, self.theta_int):
                        print('Pos: %.5s, Theta: %.5s, Qs: %s' % (p, t, self.get_qvals([p,0,t,0])))
            print('--- Q Map End ---')


class Replay_Memory():

    def __init__(self, memory_size=1000000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.term_mem_ln = 10
        self.memory_size = memory_size
        self.burn_in     = burn_in
        self.memory      = collections.deque(maxlen=self.memory_size)
        self.term_memory = collections.deque(maxlen=self.term_mem_ln)

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return [self.memory[i] for i in np.random.choice(len(self.memory), batch_size)]
        # if np.random.uniform() < 0.1:
        #     ret_val = [self.memory[i] for i in np.random.choice(len(self.memory), batch_size-1)] + \
        #               [self.term_memory[i] for i in np.random.choice(len(self.term_memory), 1)]
        #     random.shuffle(ret_val)
        # else: ret_val = [self.memory[i] for i in np.random.choice(len(self.memory), batch_size)]
        # return ret_val

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)
        # if transition[4] == True:
        #     self.term_memory.append(transition)

    def clear(self):
        self.memory.clear()
        # self.term_memory.clear()

class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #       (a) Epsilon Greedy Policy.
    #       (b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, env_name, qnmodel):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.

        self.env_name = env_name
        self.qnmodel  = qnmodel
        self.dec_eps  = 0.0000045
        self.min_eps  = 0.05
        self.q_net    = QNetwork(self.env_name, self.qnmodel)
        self.rep_mem  = Replay_Memory()

    def random_policy(self):
        # Creating random probabilities to sample from.
        return np.random.random_integers(0, self.q_net.num_act-1)

    def epsilon_greedy_policy(self, q_values, eps):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < eps: return np.random.random_integers(0, self.q_net.num_act-1)
        else:                         return self.greedy_policy(q_values)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train(self, rmem):
        # In this function, we will train our network.
        # If training without experience replay_memory, then you will interact with the environment
        # in this function, while also updating your network parameters.

        # If you are using a replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.

        # print('Train::Start')

        tick       = 0
        eps        = 0.5
        max_brn_at = 20
        test_int   = 10000
        avg_rew    = []
        do_test    = False

        if self.env_name == 'MountainCar-v0':
            gamma = 1

            max_ep_len = 200
            max_ep     = 3000

            save_int = 1000

            # record weights when rewards higher than -160 are seen
            # break out of training when rewards greater than -110 so as to save time from
            # pointless optimization
            rew_rec_thresh = -160
            rew_brk_thresh = -110
            rew_base       = -200

            # Decrease learning rate if last 3 avg rewards in test are within 30 range of
            # each other
            lr_plat_r      = 30

        if self.env_name == 'CartPole-v0':
            gamma = 0.99

            max_ep_len = 10000
            max_ep     = 100

            save_int = 33

            # record weights when rewards higher than 180 are seen
            # break out of training when rewards greater than 210 i.e. let it run to the
            # max iteration as suggested in handout. CartPole runs are pretty fast.
            rew_rec_thresh = 180
            rew_brk_thresh = 210
            rew_base       = 100

            # Decrease learning rate if last 3 avg rewards in test are within 10 range of
            # each other
            lr_plat_r      = 10

        # If goal state wasn't reached in burn-in try again. This helps for MountainCar
        # which is highly dependant on getting samples around the goal state and reaching it.
        if rmem == 'true':
            for _ in xrange(max_brn_at):
                if not self.burn_in_memory():
                    print('Train::Bad_Burn_In::No_Term_Reached')
                    self.rep_mem.clear()
                else: break

        if self.qnmodel == 'linear' and self.env_name == 'MountainCar-v0': env = gym.make(self.env_name).env
        else:                                                              env = gym.make(self.env_name)

        for ep in xrange(max_ep):
            nstate = env.reset()

            self.q_net.print_qmap()
            print('Train::Episode:(%s), LRate:%s, Eps:%s, Last Loss:%s' % (ep, self.q_net.optimizer.lr, eps, loss if 'loss' in locals() else None))

            # Roam according to epsilon greedy policy and sample observations. Put observation into replay
            # memory if required
            for _ in xrange(max_ep_len):
                pstate = nstate
                action = self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), eps)
                nstate, rew, term, info = env.step(action)
                if rmem == 'false': self.rep_mem.clear()
                self.rep_mem.append((pstate, action, rew, nstate, term))

                # Sample the last observation (just like online learning), if replay memory is switched off
                # Else sample 32
                state_batch, q_batch = [], []
                for lpst, lact, lrew, lnst, lterm in self.rep_mem.sample_batch(32 if rmem == 'true' else 1):
                    y_n       = self.q_net.get_qvals(lpst)
                    y_n[lact] = lrew if lterm else lrew+gamma*np.max(self.q_net.get_qvals(lnst))
                    q_batch.append(y_n)
                    state_batch.append(lpst)

                loss = self.q_net.update_net(state_batch, q_batch)

                # If 10000 updates have been made run test with 20 episodes
                if tick % test_int == 0: do_test = True

                tick += 1
                eps   = max(self.min_eps, eps-self.dec_eps)

                if term:
                    # print('Train::Term_state(%s)' % nstate)
                    if self.env_name == 'MountainCar-v0': break
                    if self.env_name == 'CartPole-v0': nstate = env.reset()

            if do_test: avg_rew.append(self.test())

            if (avg_rew and avg_rew[-1] >= rew_rec_thresh) or (ep % save_int == 0):
                self.q_net.save_model_weights(ep)

        self.q_net.save_model_weights(ep)
        with open(os.path.join('./model/', 'avg_rew.dat'), 'wb') as f:
            np.save(f, avg_rew)

        env.close()

        print('Train::Return')


    def test(self, num_ep=20):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        print('Test::Start')

        max_ep_len = 200
        max_ep     = num_ep
        avg_rew    = []

        env = gym.make(self.env_name).env

        for _ in xrange(max_ep):
            nstate  = env.reset()
            cur_rew = 0

            for _ in xrange(max_ep_len):
                action = self.epsilon_greedy_policy(self.q_net.get_qvals(nstate), self.min_eps)
                nstate, rew, term, _ = env.step(action)
                cur_rew += rew

                if term: break

            avg_rew.append(cur_rew)


        print('Test::Avg_Rew:%s' % np.mean(avg_rew))
        print('Test::Std_Dev:%s' % np.std(avg_rew))

        env.close()
        return np.mean(avg_rew)

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of transitions.
        print('Burn_in_memory::Start')

        sterm  = False
        env    = gym.make(self.env_name).env
        nstate = env.reset()

        for _ in xrange(self.rep_mem.burn_in):
            pstate               = nstate
            action               = self.random_policy()
            nstate, rew, term, _ = env.step(action)

            self.rep_mem.append((pstate, action, rew, nstate, term))
            if term:
                # print('Burn_in_memory::Term_State(%s, %s, %s, %s, %s)' % (pstate, action, rew, nstate, term))
                sterm = True
                nstate = env.reset()

        env.close()

        print('Burn_in_memory::Return')

        return sterm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--qnmodel',dest='qnmodel',type=str)
    parser.add_argument('--mode',dest='mode',type=str)
    parser.add_argument('--wfile',dest='wfile',type=str)
    parser.add_argument('--rmem',dest='rmem',type=str)
    return parser.parse_args()

def main(args):
    args      = parse_arguments()
    env       = args.env
    qnmodel   = args.qnmodel
    mode      = args.mode
    wfile     = args.wfile
    rmem      = args.rmem
    dqn_agent = DQN_Agent(env, qnmodel)

    if mode == 'train':
        dqn_agent.train(rmem)

    if mode == 'test':
        dqn_agent.q_net.load_model_weights(wfile)
        dqn_agent.test(100)


if __name__ == '__main__':
    main(sys.argv)

