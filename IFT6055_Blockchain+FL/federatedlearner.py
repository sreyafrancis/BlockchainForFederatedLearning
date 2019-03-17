"""
 - Blockchain for Federated Learning -
      Federated Learning Script
"""

import tensorflow as tf
import numpy as np
import pickle

def reset():
    tf.reset_default_graph()

#This function is to be called in blockchain.py after completion of each round of mining

def compute_global_model(base,updates,lrate):

    '''
    Function to get the model trainable_parameter values
    '''

    upd = dict()
    for x in ['w1','w2','wo','b1','b2','bo']:
        upd[x] = np.array(base[x], copy=True)
    number_of_clients = len(updates)
    for client in updates.keys():
        for x in ['w1','w2','wo','b1','b2','bo']:
            model = updates[client].update
            upd[x] += (lrate/number_of_clients)*(model[x]+base[x])
    upd["size"] = 0
    reset()
    dataset = dataext.load_data("data/mnist.d")
    worker = NNWorker(None,
        None,
        dataset['test_images'],
        dataset['test_labels'],
        0,
        "validation")
    worker.build(upd)
    accuracy = worker.evaluate()
    worker.close()
    return accuracy,upd

class NNWorker:
    def __init__(self,X=None,Y=None,tX=None,tY=None,size=0,id="nn0",steps=10):

        ''' 
        Function to intialize Data and Network parameters
        '''
        
        self.id = id
        self.train_x = X
        self.train_y = Y
        self.test_x = tX
        self.test_y = tY
        self.size = size
        self.learning_rate = 0.1
        self.num_steps = steps
        self.n_hidden_1 = 256 
        self.n_hidden_2 = 256 
        self.num_input = 784
        self.num_classes = 10 
        self.sess = tf.Session()

    def build(self,base):

        ''' 
        Function to initialize/build network based on updated values received 
        from blockchain
        '''
        
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.weights = {
            'w1': tf.Variable(base['w1'],name="w1"),
            'w2': tf.Variable(base['w2'],name="w2"),
            'wo': tf.Variable(base['wo'],name="wo")
        }
        self.biases = {
            'b1': tf.Variable(base['b1'],name="b1"),
            'b2': tf.Variable(base['b2'],name="b2"),
            'bo': tf.Variable(base['bo'],name="bo")
        }
        
        self.layer_1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1'])
        self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['w2']),self.biases['b2'])
        self.out_layer = tf.matmul(self.layer_2, self.weights['wo'])+self.biases['bo']
        self.logits = self.out_layer
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def build_base(self):

        ''' 
        Function to initialize/build network with random initialization
        '''
        
        self.X = tf.placeholder("float", [None, self.num_input])
        self.Y = tf.placeholder("float", [None, self.num_classes])
        self.weights = {
            'w1': tf.Variable(tf.random_normal([self.num_input, self.n_hidden_1]),name="w1"),
            'w2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]),name="w2"),
            'wo': tf.Variable(tf.random_normal([self.n_hidden_2, self.num_classes]),name="wo")
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1]),name="b1"),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2]),name="b2"),
            'bo': tf.Variable(tf.random_normal([self.num_classes]),name="bo")
        }
        
        self.layer_1 = tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1'])
        self.layer_2 = tf.add(tf.matmul(self.layer_1, self.weights['w2']),self.biases['b2'])
        self.out_layer = tf.matmul(self.layer_2, self.weights['wo'])+self.biases['bo']
        self.logits = self.out_layer
        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):

        ''' 
        Function to train the data, optimize and calculate loss and accuracy per batch
        '''
        
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        for step in range(1, self.num_steps+1):
            
            self.sess.run(self.train_op, feed_dict={self.X: self.train_x, self.Y: self.train_y})
            loss, acc = self.sess.run([self.loss_op, self.accuracy],
            feed_dict={self.X: self.train_x,self.Y: self.train_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                 "{:.4f}".format(loss) + ", Training Accuracy= " + \
                 "{:.3f}".format(acc))
            print("Optimization Finished!")


    def centralized_accuracy(self):

        ''' 
        Function to train the data and calculate centralized accuracy based on 
        evaluating the updated model peformance on test data 
        '''
        cntz_acc = dict()
        cntz_acc['epoch'] = []
        cntz_acc['accuracy'] = []

        self.build_base()
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        
        for step in range(1,self.num_steps+1):
            self.sess.run(self.train_op, feed_dict={self.X: self.train_x, self.Y: self.train_y})
            cntz_acc['epoch'].append(step)
            acc = self.evaluate()
            cntz_acc['accuracy'].append(acc)
            print("epoch",step,"accuracy",acc)
        return cntz_acc


    def evaluate(self):

        '''
        Function to calculate accuracy on test data
        '''
        return self.sess.run(self.accuracy, feed_dict={self.X: self.test_x,self.Y:self.test_y})


    def get_model(self):

        '''
        Function to get the model trainable_parameter values
        '''
        varsk = {tf.trainable_variables()[i].name[:2]:tf.trainable_variables()[i].eval(self.sess) for i in range(len(tf.trainable_variables()))}
        varsk["size"] = self.size
        return varsk

    def close(self):

        '''
        Function to close the current session
        '''
        self.sess.close()


