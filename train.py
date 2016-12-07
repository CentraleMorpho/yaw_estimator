import vgg as model
import tensorflow as tf
import numpy as np
import time
import pickle


def train(lr=0.0001,
          nb_iterations=10000,
          batch_size=64):


    with tf.Graph().as_default():

	#images, paths = model.distorted_inputs('training', batch_size, dico)
 	images, labels = model.distorted_inputs('training', batch_size)
	imagesVal, labelsVal = model.distorted_inputs('test', batch_size)

        #labels = tf.placeholder("float", [batch_size, 3])
        logits = model.inference_cifar10_vgg(images, training=True)
        objective = model.loss_op(logits, labels, batch_size)
	accuracy = model.evaluate_op(logits, labels)
        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective, global_step=global_step)

	logitsVal = model.inference_cifar10_vgg(imagesVal, training=False)
	accuracyVal = model.evaluate_op(logitsVal, labelsVal)
	


        # Start running operations on the Graph.
        with tf.Session() as sess:
	    saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
	    sess.run(tf.initialize_local_variables())
	    #saver.restore(sess,"model.cpkt")
	    #print("Model restored")

	    coord = tf.train.Coordinator()
    	    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	    model.init()


            for iteration in range(nb_iterations):
  
		    t0 = time.time()
		    

		    #result1 = sess.run([images, paths])
		    #imagesEval = result1[0]
		    #pathsEval = result1[1]

 
		    #result = sess.run(
			#[ train_step, objective, accuracy, logits],
			#	feed_dict={images:imagesEval, labels:labelsEval}
		    #)  

		    #result = sess.run(
			#[ images, labels, train_step, objective, accuracy, logits]
		    #)  

		    result = sess.run(
			[images, labels, train_step, objective, accuracy, logits], 
		    ) 
		    trn_loss = result[3]
		    trn_acc = result[4]
		    logitsArray = result[5]
		    duration = time.time() - t0



		    # print debugging info
		    print("iter:%5d, trn_loss: %s, precisions YPR : %s, %s, %s" % (iteration, trn_loss, "%.2f" % trn_acc[0], "%.2f" % trn_acc[1], "%.2f" % trn_acc[2]))

		    if iteration % 10 == 0:

		        t0 = time.time()

			result = sess.run(
				[imagesVal, labelsVal, train_step,logitsVal,accuracyVal], 
		        ) 

		   	trn_acc = result[4]
			print("iter:%5d, VALIDATION BATCH, precisions YPR : %s, %s, %s" % (iteration, trn_acc[0], trn_acc[1], trn_acc[2]))
            
	    print("Saving model...")
	    save_path=saver.save(sess, "model.cpkt")
	    print("Model saved in file : %s" % save_path)
			
                    


if __name__ == '__main__':
    batch_size = 10
    train(batch_size = batch_size)
