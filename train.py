import vgg as model
import tensorflow as tf
import numpy as np
import time
import firstscript


def train(data, dataLabels, lr=0.0001,
          nb_epochs=10,
          batch_size=12):


    with tf.Graph().as_default():

        images = tf.placeholder("float", [batch_size, 39, 39,1])
        labels = tf.placeholder("float", [batch_size, 3])
        logits = model.inference_cifar10_vgg(images, training=True)
        # predictions, softmax, logits = model.inference_op(images, training=True)
        objective = model.loss_op(logits, labels, batch_size)
        #approx_correct = model.evaluate_op(logits, labels)
	accuracy = model.evaluate_op(logits, labels)
        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective, global_step=global_step)

        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())


            for epoch in range(nb_epochs):

                for step in range(3200/batch_size):
                    # get batch and format data
                    X,Y = getBatch(data, dataLabels, step, batch_size)
		   
             
		    X=X.reshape((batch_size, 39,39,1))
                   


                    t0 = time.time()
                    result = sess.run(
			[train_step, objective, accuracy, logits],
                        feed_dict = {
                            images: X,
                            labels: Y,
                        }
                    )
                    trn_loss = result[1]
                    trn_acc = result[2]
		    logitsArray = result[3]
		    #print(logitsArray)
                    duration = time.time() - t0


                    # print debugging info
                    print("epoch:%5d, step:%5d, trn_loss: %s, precisions YPR : %s, %s, %s" % (epoch, step, trn_loss, trn_acc[0], trn_acc[1], trn_acc[2]))
                    


def getBatch(data, dataLabels, step, batch_size):
	X = data[step*batch_size:(step+1)*batch_size,:,:]
	Y = dataLabels[step*batch_size:(step+1)*batch_size,:]
	return [X,Y]
                 

if __name__ == '__main__':
    import dataset
    batch_size = 20
    folder = '../Cropped/img_014b/GeneratedImgs/014b/Campagne_IR_Pose_20150710/cam1/1'


    data = firstscript.load_dataset(folder)
    dataLabels = firstscript.findLabelsMatrix(folder)
    train(data, dataLabels,batch_size=batch_size)
