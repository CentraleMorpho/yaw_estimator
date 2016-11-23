import vgg as model
import tensorflow as tf
import numpy as np
import time
import pickle


def train(lr=0.0001,
          nb_iterations=10000,
          batch_size=12):

    with open('dictLabels.pkl','rb') as f:
	dico = pickle.load(f)


    with tf.Graph().as_default():

	images, paths = model.distorted_inputs('training', batch_size)
	imagesVal, pathsVal = model.distorted_inputs('test', batch_size)

        labels = tf.placeholder("float", [batch_size, 3])
        logits = model.inference_cifar10_vgg(images, training=True)
        objective = model.loss_op(logits, labels, batch_size)
	accuracy = model.evaluate_op(logits, labels)
        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective, global_step=global_step)
	


        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
	    sess.run(tf.initialize_local_variables())

	    coord = tf.train.Coordinator()
    	    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


            for iteration in range(nb_iterations):
  
		    t0 = time.time()
		    

		    result1 = sess.run([images, paths])
		    imagesEval = result1[0]
		    pathsEval = result1[1]

		    labelsEval = np.zeros([batch_size,3], dtype=float)
		    for i in range(0,len(pathsEval)):
			labelsEval[i,:]=dico[pathsEval[i]]

 
		    result = sess.run(
			[ train_step, objective, accuracy, logits],
				feed_dict={images:imagesEval, labels:labelsEval}
		    )    
		    trn_loss = result[1]
		    trn_acc = result[2]
		    logitsArray = result[3]
		    duration = time.time() - t0



		    # print debugging info
		    print("iter:%5d, trn_loss: %s, precisions YPR : %s, %s, %s" % (iteration, trn_loss, trn_acc[0], trn_acc[1], trn_acc[2]))

		    if iteration % 10 == 0:

		        t0 = time.time()

			result1 = sess.run([imagesVal, pathsVal])
			imagesValEval = result1[0]
			pathsValEval = result1[1]

			labelsValEval = np.zeros([batch_size,3], dtype=float)
			for i in range(0,len(pathsValEval)):
				labelsValEval[i,:]=dico[pathsValEval[i]]

		    	result = sess.run(
				[accuracy], feed_dict={images:imagesValEval, labels:labelsValEval}
		    	)
		   	trn_acc = result[0]
			print("iter:%5d, VALIDATION BATCH, precisions YPR : %s, %s, %s" % (iteration, trn_acc[0], trn_acc[1], trn_acc[2]))
			
                    


if __name__ == '__main__':
    batch_size = 20
    train(batch_size = batch_size)
