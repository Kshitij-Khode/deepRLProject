from QNetwork     import *
from ReplayMemory import *
from skimage      import color, transform

class DQNAgent():

    def __init__(
        self,
        qnModel,
        initLRate,
        optEps,
        picStackLen,
        memSize,
        picsize,
        batchSize
    ):

        self.qNet = QNetwork()
        self.repMem = ReplayMemory(picStackLen, memSize, picsize)

        self.actMap = None
        self.actDim = 0
        self.picsize = picsize
        self.batchSize = batchSize
        self.optParams = dict(learning_rate=initLRate, epsilon=optEps)

    def buildGraph(self):
        inBatchDim = (self.batchSize, self.pickStackLen) + self.picSize
        inGenDim = (None, self.pickStackLen) + self.picSize

        self.inPrevState = tf.placeholder(tf.float32, inGenDim, "prevState")
        self.inNextState = tf.placeholder(tf.float32, inBatchDim, "nextState")
        self.inRew = tf.placeholder(tf.float32, self.batchSize, "reward")
        self.inAct = tf.placeholder(tf.int32, self.batchSize, "actions")
        self.inDoneMask = tf.placeholder(tf.int32, self.batchSize, "doneMask")

        with tf.variable_scope("fixed"):
            qsaTargets = self.create_network(self.inNextState, trainable=False)
        with tf.variable_scope("train"):
            qsaEst = self.create_network(self.inPrevState, trainable=True)

        self.bestAct = tf.argmax(qsaEst, axis=1)

        notDone = tf.cast(tf.logical_not(tf.cast(self.inDoneMask, "bool")), "float32")
        qTarget = tf.reduce_max(qsaTargets, -1) * self.gamma * notDone + self.inRew

        actSlice = tf.stack([tf.range(0, self.batchSize), self.inAct], axis=1)
        qEstInAct = tf.gather_nd(qsaEst, actSlice)

        trainLoss = tf.nn.l2_loss(qTarget-qEstInAct) / self.batchSize

        optimizer = tf.train.AdamOptimizer(**(self.optParams))

        regLoss = tf.add_n(tf.losses.get_regularization_losses())
        self.train_op = optimizer.minimize(regLoss+trainLoss)

        trainParams = self.get_variables("train")
        fixedParams = self.get_variables("fixed")

        assert (len(trainParams) == len(fixedParams))
        self.copyNetOps = [tf.assign(fixedV, trainV)
            for trainV, fixedV in zip(trainParams, fixedParams)]

    def get_variables(self, scope):
        vars = [t for t in tf.global_variables()
            if "%s/" % scope in t.name and "Adam" not in t.name]
        return sorted(vars, key=lambda v: v.name)

    def create_network(self, input, trainable):
        if trainable:
            wr = slim.l2_regularizer(self.regularization)
        else:
            wr = None

        # the input is stack of black and white frames.
        # put the stack in the place of channel (last in tf)
        input_t = tf.transpose(input, [0, 2, 3, 1])

        net = slim.conv2d(input_t, 8, (7, 7), data_format="NHWC",
            activation_fn=tf.nn.relu, stride=3, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.conv2d(net, 16, (3, 3), data_format="NHWC",
            activation_fn=tf.nn.relu, weights_regularizer=wr, trainable=trainable)
        net = slim.max_pool2d(net, 2, 2)
        net = slim.flatten(net)
        # net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
        #     weights_regularizer=wr, trainable=trainable)
        # q_state_action_values = slim.fully_connected(net, self.dim_actions,
        #     activation_fn=None, weights_regularizer=wr, trainable=trainable)

        netv = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
            weights_regularizer=wr, trainable=trainable)
        neta = slim.fully_connected(net, 256, activation_fn=tf.nn.relu,
            weights_regularizer=wr, trainable=trainable)
        netv = slim.fully_connected(netv, 1, activation_fn=None,
            weights_regularizer=wr, trainable=trainable)
        neta = slim.fully_connected(neta, self.dim_actions, activation_fn=None,
            weights_regularizer=wr, trainable=trainable)

        q_state_action_values = netv + (self.neta - tf.reduce_mean(self.neta, reduction_indices=[1,], keep_dims=True))


        return q_state_action_values

    def check_early_stop(self, reward, totalreward):
        return False, 0.0

    def get_random_action(self):
        return np.random.choice(self.dim_actions)

    def get_epsilon(self):
        if not self.do_training:
            return self.playing_epsilon
        elif self.global_counter >= self.epsilon_decay_steps:
            return self.min_epsilon
        else:
            # linear decay
            r = 1.0 - self.global_counter / float(self.epsilon_decay_steps)
            return self.min_epsilon + (self.initial_epsilon - self.min_epsilon) * r

    def train(self):
        batch = self.exp_history.sample_mini_batch(self.batchsize)

        fd = {
            self.input_reward: "reward",
            self.input_prev_state: "prev_state",
            self.input_next_state: "next_state",
            self.input_actions: "actions",
            self.input_done_mask: "done_mask"
        }
        fd1 = {ph: batch[k] for ph, k in fd.items()}
        self.session.run([self.train_op], fd1)

    def play_episode(self):
        eh = (
            self.exp_history if self.do_training
            else self.playing_cache
        )
        total_reward = 0
        frames_in_episode = 0

        first_frame = self.env.reset()
        first_frame_pp = self.process_image(first_frame)

        eh.start_new_episode(first_frame_pp)

        while True:
            if np.random.rand() > self.get_epsilon():
                action_idx = self.session.run(
                    self.bestAct,
                    {self.input_prev_state: eh.current_state()[np.newaxis, ...]}
                )[0]
            else:
                action_idx = self.get_random_action()

            if self.action_map is not None:
                action = self.action_map[action_idx]
            else:
                action = action_idx

            reward = 0
            for _ in range(self.frame_skip):
                observation, r, done, info = self.env.step(action)
                if self.render:
                    self.env.render()
                reward += r
                if done:
                    break

            early_done, punishment = self.check_early_stop(reward, total_reward)
            if early_done:
                reward += punishment

            done = done or early_done

            total_reward += reward
            frames_in_episode += 1

            eh.add_experience(self.process_image(observation), action_idx, done, reward)

            if self.do_training:
                self.global_counter += 1
                if self.global_counter % self.network_update_freq:
                    self.update_target_network()
                train_cond = (
                    self.exp_history.counter >= self.min_experience_size and
                    self.global_counter % self.train_freq == 0
                )
                if train_cond:
                    self.train()

            if done:
                if self.do_training:
                    self.episode_counter += 1

                return total_reward, frames_in_episode

    def update_target_network(self):
        self.session.run(self.copyNetOps)

class CarRacingDQNAgent(DQNAgent):

    self.negRewCount = 0
    self.maxNegRew = 12
    self.gasActs = None

    super().__init__(**kwargs)
    initActionSpace()

    def process_image(obs):
        return 2 * color.rgb2gray(obs) - 1.0

    def initActionSpace():
        self.actMap = np.array([k for k in it.product([-1, 0, 1], [1, 0], [0.2, 0])])
        self.actDim = gym.make('CarRacing-v0').action_space.n
        self.gasActs = np.array([a[1] == 1 and a[2] == 0 for a in self.actMap])

    def getRandomAction(self):
        actW  = 14.0 * self.gasActs + 1.0
        actW /= np.sum(actW)

        return np.random.choice(self.dim_actions, p=actW)

    def checkEarlyStop(self, rew, tRew):
        if rew < 0:
            self.negRewCount += 1
            done = (self.negRewCount > self.maxNegRew)

            if done and tRew <= 500: punishment = -20.0
            else: punishment = 0.0
            if done: self.negRewCount = 0

            return done, punishment

        else:
            self.negRewCount = 0
            return False, 0.0

