import os
import numpy as np
import tensorflow as tf
import time
import collections
import random

class DeepQNetwork(object):

    def __init__(self, lr, n_actions, name, fc1_dims=512, LSTM_DIM=256,
                 input_dims=(210, 160, 4), chkpt_dir="tmp/dqn"):
        self.lr = lr
        self.name = name
        self.LSTM_DIM = LSTM_DIM
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=None)
        self.checkpoint_file = os.path.join(chkpt_dir, "deepqnet.ckpt")
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope=self.name)
        self.write_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("tmp/log_dir", self.sess.graph)
        #self.writer.add_graph(self.sess.graph)

    def build_network(self):

        with tf.variable_scope(self.name):
            self.states = tf.placeholder(tf.float32, shape=[None, *self.input_dims],
                                        name='states')
           
            # * here indicates that the function can take multiple inputs as arguments into the function.
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                          name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None],
                                           name='q_value')

            self.seq_len = tf.placeholder(tf.int32, name='sequence_length')
            self.batch_size = tf.placeholder(tf.int32, name='batch_size')

            # Create placeholders to input the hidden state values
            c_in = tf.placeholder(tf.float32, [None, self.LSTM_DIM], name='cell_state')
            h_in = tf.placeholder(tf.float32, [None, self.LSTM_DIM], name='h_state')
            self.state_in = tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in)


            self._reward = tf.placeholder(tf.float32, shape=[], name='Reward/Time_step')
            self.reward_sum = tf.summary.scalar('Reward/Time_step', self._reward)

            self._waitingtime = tf.placeholder(tf.float32, shape=[], name='TotalWaitingTime/Time_step')
            self.waitingtime_sum = tf.summary.scalar('TotalWaitingTime/Time_step', self._waitingtime)

            self._delay = tf.placeholder(tf.float32, shape=[], name='TotalDelay/Time_step')
            self.delay_sum = tf.summary.scalar('TotalDelay/Time_step', self._delay)

            # 1st dimension inside shape is set to None because we want to pass
            # batches of stacked frame into the neural network.

            conv1 = tf.layers.conv2d(inputs=self.states, filters=32,
                                     kernel_size=(8, 8), strides=4, name='conv1',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))
            # TensorShape([Dimension(None), Dimension(44), Dimension(39), Dimension(32)])

            conv1_activated = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                     kernel_size=(4, 4), strides=2, name='conv2',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))
            # TensorShape([Dimension(None), Dimension(21), Dimension(18), Dimension(64)])

            conv2_activated = tf.nn.relu(conv2)

            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=64,
                                     kernel_size=(3, 3), strides=1, name='conv3',
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))

            conv3_activated = tf.nn.relu(conv3)

            n_input = conv3_activated.get_shape().as_list()[1]*conv3_activated.get_shape().as_list()[2]*conv3_activated.get_shape().as_list()[3]
            
            conv3_activated = tf.reshape(conv3_activated, [-1, n_input])
            
            conv3_activated = tf.reshape(conv3_activated, [self.batch_size, self.seq_len, n_input])
   
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.LSTM_DIM, initializer=tf.contrib.layers.xavier_initializer())
            outputs, state = tf.nn.dynamic_rnn(lstm_cell, conv3_activated, initial_state=self.state_in, dtype=tf.float32, sequence_length=self.seq_len)

            var1 = tf.get_variable('weights', (self.LSTM_DIM, self.n_actions), initializer=tf.contrib.layers.xavier_initializer(), 
                                                    regularizer=tf.contrib.layers.l2_regularizer(0.0))
            var2 = tf.get_variable('biases', (self.n_actions,), initializer=tf.constant_initializer(0.1))

            h = outputs[:,-1,:] 

            self.Q_values = tf.matmul(h, var1) + var2
            tf.summary.histogram('Q_value', self.Q_values)
            ''' Add softmax layer for probability distribution.'''


            '''self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                                           kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))'''

            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions), axis=1)
            #self.accuracy_sum = tf.summary.scalar('Accuracy', self.q)

            self.loss = tf.reduce_mean(tf.square(self.q_target - self.q))
            self.loss_sum = tf.summary.scalar("Loss", self.loss)

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            if self.name == 'q_eval':
                for var in tf.trainable_variables():
                    c = var.name[:-2]
                    with tf.name_scope(c):
                        self.variable_summaries(var)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self, epi_num):
        print('... Saving Checkpoint ...')
        self.epi_num = epi_num
        dir_name = os.path.join(self.chkpt_dir, str(self.epi_num))
        os.mkdir(dir_name)
        filename = "deepQnet_" + str(epi_num) + ".ckpt"
        self.checkpoint_file = os.path.join(dir_name, filename)
        self.saver.save(self.sess, self.checkpoint_file) 

class Agent(object):
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
                 replace_target=30000, input_dims=(210, 160, 4), q_next_dir="tmp/q_next/single/cross", q_eval_dir="tmp/q_eval/single/cross"):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        # for n_actions=2, action_space is a list [0, 1]
        self.gamma = gamma
        self.input_dims = input_dims
        self.LSTM_DIM = 256
        self.seq_length = 5
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_eval', chkpt_dir=q_eval_dir)

        self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                   name='q_next', chkpt_dir=q_next_dir)

        # Defining arrays of zeros for state, rewards etc to be stored.
        self.state_memory = np.zeros((self.mem_size, *input_dims))
        # will save a set of 4 stacked frames by number of memories.
        # state_memory has shape(mem_size, 210, 160, 4).

        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions),
                                      dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
        # terminal only has 0 and 1 as input. At end of an episode
        # shape of terminal_memory is (mem_size, 0) with all elements being 0.
        # we don't to have future rewards, this will be indicated by the terminal memory.


    def store_transition(self, state, action, reward, state_, terminal):

        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.reward = reward
        actions = np.zeros(self.n_actions)
        # actions are arrays of zeros(vector) with len equal to the number of actions. [0, 0, 0]
        # action_space is a list of len equal to the  number of action [0, 1, 2]

        for x, y in action.items():
            actions[y] = 1.0

        self.action_memory[index] = actions
        self.reward_memory[index] = reward.get('result')
        # shape (mem_size,)

        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal
        # It is a done flag, 1 for done(True) and 0 for not done(False).
        # It indicates the completion of an episode.

        self.mem_cntr += 1

        if self.mem_cntr > self.mem_size:
            self.epsilon = 0.01

    def choose_action(self, state, epsilon):
        rand = np.random.random()
        state = (np.zeros((1, self.LSTM_DIM)),np.zeros((1, self.LSTM_DIM)))
        if rand < self.epsilon:
            act = np.random.choice(self.action_space)
            # since action_space has actions as [0, 1], it will generate an integer from elements of action space.
            # That is either 0th, 1st or 2nd action.

            action = collections.OrderedDict()
            action['0'] = act
            # Here zero is the traffic Light ID. Need to make it more robust for multli agent.

        else:
            act = self.q_eval.sess.run(self.q_eval.Q_values,
                                       feed_dict={self.q_eval.states: state,
                                                  self.q_eval.seq_len: self.seq_length,
                                                  self.q_eval.batch_size: 1})
            act = np.argmax(act)

            action = collections.OrderedDict()
            action['0'] = act

        return action

    def RandomSequenceSampling(self):
        batch_length = self.batch_size*self.seq_length
        state_batch = np.zeros((batch_length, *self.input_dims))
        next_state_batch = np.zeros((batch_length, *self.input_dims))
        reward_batch = []
        action_batch = []
        terminal_batch = []

        indices = np.arange(self.seq_length-1, self.mem_size)

        for b in np.arange(0, batch_length, self.seq_length):
            i = random.choice(indices)
            while (sum(self.terminal_memory[i+1-self.seq_length:i+1]) > 0 and self.terminal_memory[i] != 1):
                i = random.choice(indices)
            state_batch[b:b+self.seq_length] = self.get_sequence(i, self.state_memory)
            action_batch.append(self.action_memory[i])
            reward_batch.append(self.reward_memory[i])
            next_state_batch[b:b+self.seq_length] = self.get_sequence(i, self.new_state_memory)
            terminal_batch.append(self.terminal_memory[i])

        return state_batch, np.asarray(action_batch), np.asarray(reward_batch), next_state_batch, np.asarray(terminal_batch)

    def get_sequence(self, index, collection):
        stop = index + 1 
        start = stop - self.seq_length
        
        if start < 0 and stop >= 0:
            try:
                seq = np.vstack((collection[start:], collection[:stop]))
            except ValueError:
                seq = np.append(collection[start:], collection[:stop])
        else:
            seq = collection[start:stop]

        # The append operation adds an extra dimension to the matrix
        if len(seq.shape) != len(collection.shape):
            seq = np.reshape(seq, (-1,))

        return seq

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()
        # we update the graph after every K steps, so that the q_target is not fluctuating.

        ''' max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size

        batch = np.random.choice(max_mem, self.batch_size)
        # It can be replaced with random.sample().
        # Batch is of the length equal to batch size with elements that are generated using np.arange(max_mem).

        state_batch = self.state_memory[batch]

        action_batch = self.action_memory[batch]

        action_values = np.array([0, 1], dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)

        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        terminal_batch = self.terminal_memory[batch]'''
    
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.RandomSequenceSampling() 

        state = (np.zeros((self.batch_size, self.LSTM_DIM)),np.zeros((self.batch_size, self.LSTM_DIM)))

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                      feed_dict={self.q_eval.states: state_batch,
                                                 self.q_eval.state_in: state,
                                                 self.q_eval.seq_len: self.seq_length,
                                                 self.q_eval.batch_size: self.batch_size})

        # It has shape (batch_size, n_actions).
        # This gives Q values for each action, in this case 3 actions, using q_eval network for current state batch.

        q_eval_next = self.q_eval.sess.run(self.q_eval.Q_values,
                                           feed_dict={self.q_eval.states: next_state_batch,
                                                      self.q_eval.state_in: state,
                                                      self.q_eval.seq_len: self.seq_length,
                                                      self.q_eval.batch_size: self.batch_size})

        index_best_action = np.argmax(q_eval_next, axis=1)

        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                      feed_dict={self.q_next.states: next_state_batch,
                                                 self.q_next.state_in: state,
                                                 self.q_next.seq_len: self.seq_length,
                                                 self.q_next.batch_size: self.batch_size})

        # This gives Q values for the next state using the q_next network.

        idx = np.arange(self.batch_size)
        q_target = reward_batch + \
            self.gamma*(q_next[idx, index_best_action])*(1 - terminal_batch)

     
        _, summary1 = self.q_eval.sess.run([self.q_eval.train_op, self.q_eval.write_op],
                                 feed_dict={self.q_eval.states: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target,
                                            self.q_eval.seq_len: self.seq_length,
                                            self.q_eval.batch_size: self.batch_size,
                                            self.q_eval.state_in: state,
                                            self.q_eval._reward: self.reward['result'],
                                            self.q_eval._waitingtime: self.reward['total_waiting'],
                                            self.q_eval._delay: self.reward['total_delay']})

        '''summary1 = self.q_eval.sess.run(self.q_eval.write_op,
                                        feed_dict={self.q_eval.states: state_batch,
                                                   self.q_eval.actions: action_batch,
                                                   self.q_eval.seq_len: self.seq_length,
                                                   self.q_eval.q_target: q_target,
                                                   self.q_eval._reward: self.reward['result'],
                                                   self.q_eval._waitingtime: self.reward['total_waiting'],
                                                   self.q_eval._delay: self.reward['total_delay']})'''

        self.q_eval.writer.add_summary(summary1, self.mem_cntr)
        self.q_eval.writer.flush()


    def test(self, epsilon):
        

    def save_models(self, episode_number):
        self.episode_number = episode_number
        self.q_eval.save_checkpoint(epi_num = self.episode_number)
        self.q_next.save_checkpoint(epi_num = self.episode_number)

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def update_graph(self):
        t_params = self.q_next.params
        e_params = self.q_eval.params

        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t, e))


