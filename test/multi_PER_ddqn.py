import os
import numpy as np
import tensorflow as tf
import time
import collections
import itertools as it
import random

class DeepQNetwork(object):

    def __init__(self, lr, n_actions, name, fc1_dims=512, #dirname,
                 input_dims=(210, 160, 4), chkpt_dir="tmp/dqn"):
        self.lr = lr
        self.name = name
        #self.dirname = dirname
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
        #self.logfile = os.path.join("tmp/log_dir/multi", str(self.dirname))
        self.writer = tf.summary.FileWriter("tmp/log_dir/multi", self.sess.graph)
        #self.writer.add_graph(self.sess.graph)

    def build_network(self):

        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims],
                                        name='inputs')
            # * here indicates that the function can take multiple inputs as arguments into the function.
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_actions],
                                          name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None],
                                           name='q_value')
            self.ISWeights_ = tf.placeholder(tf.float32, [None,1], name='IS_weights')

            self._reward = tf.placeholder(tf.float32, shape=[], name='Reward/Time_step')
            self.reward_sum = tf.summary.scalar('Reward/Time_step', self._reward)

            self._waitingtime = tf.placeholder(tf.float32, shape=[], name='TotalWaitingTime/Time_step')
            self.waitingtime_sum = tf.summary.scalar('TotalWaitingTime/Time_step', self._waitingtime)

            self._delay = tf.placeholder(tf.float32, shape=[], name='TotalDelay/Time_step')
            self.delay_sum = tf.summary.scalar('TotalDelay/Time_step', self._delay)

            # 1st dimension inside shape is set to None because we want to pass
            # batches of stacked frame into the neural network.

            conv1 = tf.layers.conv2d(inputs=self.input, filters=32,
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

            flat = tf.contrib.layers.flatten(conv3_activated)
            # A flattened tensor with shape [batch_size, k].

            dense1 = tf.layers.dense(flat, units=self.fc1_dims, activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))

            self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(factor=2))

            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions), axis=1)
        
            self.absolute_errors = tf.abs(self.q_target - self.q)            

            #self.loss = tf.reduce_mean(tf.square(self.q_target - self.q))

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.q_target, self.q))
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


class SumTree(object):
    
    data_pointer = 0
    def __init__(self, capacity):
        # Number of leaf nodes that will contain the experiences.
        self.capacity = capacity

        # Parent nodes = capacity -1
        # Leaf Nodes = capacity
        # 2*capacity -1 is the total number of leaves in a SumTree.
        self.tree = np.zeros(2 * capacity - 1)

        #contains the experience
        self.data = np.zeros(capacity, dtype=object)

    def add(self, priority, data):
        # Filling the tree from left to right
        tree_index = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        
        # Updating the tree with the priority score
        self.update (tree_index, priority)  
        self.data_pointer += 1  

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree

    PER_e = 0.01 
    PER_a = 0.6  
    PER_b = 0.4
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
         
        self.tree = SumTree(capacity)
        
    """
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
 
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Agent(object):
    def __init__(self, alpha, gamma, mem_size, epsilon, batch_size, num_agents, act_per_agent,
                 replace_target=30000, input_dims=(210, 160, 4), q_next_dir="tmp/q_next/multi/next", q_eval_dir="tmp/q_eval/multi/eval"):
        self.num_agents = num_agents
        self.act_per_agent = act_per_agent
        self.n_actions = self.act_per_agent**(self.num_agents)
        self.action_space = [i for i in range(self.act_per_agent)]
        # for n_actions=2, action_space is a list [0, 1]
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.q_eval = DeepQNetwork(alpha, self.n_actions, input_dims=input_dims,
                                   name='q_eval', chkpt_dir=q_eval_dir)

        self.q_next = DeepQNetwork(alpha, self.n_actions, input_dims=input_dims,
                                   name='q_next', chkpt_dir=q_next_dir)

        self.memory = Memory(mem_size)

        self.all_list = []
        for j in it.product(tuple(self.action_space), repeat = self.num_agents):
            self.all_list.append(j) 

    def store_transition(self, experience):
        state, action, reward, state_, terminal = experience
        action = self.action_hot_encoder(action, self.all_list)
        self.rewards = reward
        reward = reward['result']
        experience = state, action, reward, state_, terminal
        self.mem_cntr+=1
        self.memory.store(experience)

    def action_hot_encoder(self, actions, all_list):
        action = np.zeros((self.n_actions))
        value_list = tuple(actions.values())
        for key, val in enumerate(all_list):
            if val == value_list:
                action[key] = 1.
                break 
        return action

    def action_decoder(self, encoded_action, all_list):
        index = (list(np.where(encoded_action==1.))[0])[0]
        decoded_action = collections.OrderedDict()

        for i in range(len(encoded_action)):
            try:
                decoded_action[str(i)] = all_list[index][i]

            except:
                break

        return decoded_action

    def choose_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            value_list = []
            for i in range(self.num_agents):
                value_list.append(np.random.choice(self.action_space))
            value_list = tuple(value_list)
            action = np.zeros((self.n_actions))
            for key, val in enumerate(self.all_list):
                if val == value_list:
                    action[key] = 1.
                    break
            action = self.action_decoder(action, self.all_list)

            # since action_space has actions as [0, 1], it will generate an integer from elements of action space.
            # That is either 0th, 1st or 2nd action.

        else:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                       feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
            action = self.action_decoder(action, self.all_list)

        return action

    def learn(self):
        if self.mem_cntr % self.replace_target == 0:
            self.update_graph()

        tree_idx, batch, ISWeights_batch = self.memory.sample(self.batch_size)

        state_batch = np.array([each[0][0] for each in batch])
        action_batch = np.array([each[0][1] for each in batch])
        reward_batch = np.array([each[0][2] for each in batch])
        next_state_batch = np.array([each[0][3] for each in batch])
        terminal_batch = np.array([each[0][4] for each in batch])

        q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                      feed_dict={self.q_eval.input: state_batch})

        q_eval_next = self.q_eval.sess.run(self.q_eval.Q_values,
                                           feed_dict={self.q_eval.input: next_state_batch})

        index_best_action = np.argmax(q_eval_next, axis=1)

        q_next = self.q_next.sess.run(self.q_next.Q_values,
                                      feed_dict={self.q_next.input: next_state_batch})

        idx = np.arange(self.batch_size)
        q_target = reward_batch + \
            self.gamma*(q_next[idx, index_best_action])*(1 - terminal_batch)

        
        _, loss, absolute_errors = self.q_eval.sess.run([self.q_eval.train_op, self.q_eval.loss, self.q_eval.absolute_errors],
                                 feed_dict={self.q_eval.input: state_batch,
                                            self.q_eval.actions: action_batch,
                                            self.q_eval.q_target: q_target,
                                            self.q_eval.ISWeights_: ISWeights_batch})

        self.memory.batch_update(tree_idx, absolute_errors)

        summary1 = self.q_eval.sess.run(self.q_eval.write_op,
                                        feed_dict={#self.q_eval.input: state_batch,
                                                   #self.q_eval.actions: action_batch,
                                                   #self.q_eval.q_target: q_target
                                                   self.q_eval.loss: loss,
                                                   self.q_eval._reward: self.rewards['result'],
                                                   self.q_eval._waitingtime: self.rewards['total_waiting'],
                                                   self.q_eval._delay: self.rewards['total_delay']})

        
        self.q_eval.writer.add_summary(summary1)
        self.q_eval.writer.flush()

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


