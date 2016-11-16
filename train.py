import vgg as model
import tensorflow as tf
import numpy as np
import time
import firstscript


def train(data, dataLabels, lr=0.01,
          nb_epochs=10,
          batch_size=12):


    with tf.Graph().as_default():

        images = tf.placeholder("float", [batch_size, 39, 39,1])
        print images.get_shape()
        labels = tf.placeholder("float", [batch_size, 3])
        logits = model.inference_cifar10_vgg(images, training=True)
        # predictions, softmax, logits = model.inference_op(images, training=True)
        objective = model.loss_op(logits, labels, batch_size)
        #accuracy, total_correct = model.evaluate_op(logits, labels)
	accuracy = objective
        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective, global_step=global_step)

        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())


            for epoch in range(nb_epochs):

                for step in range(3200/batch_size):
                    # get batch and format data
                    batch = getBatch(data, dataLabels, step, batch_size)
                    X = np.array(batch[0])
                    Y = np.array(batch[1])
		    X=X.reshape((batch_size, 39,39,1))


                    t0 = time.time()
                    result = sess.run(
                        #[train_step, objective, accuracy, predictions],
			[train_step, accuracy, logits],
                        feed_dict = {
                            images: X,
                            labels: Y,
                        }
                    )
                    trn_loss = result[1]
                    trn_acc = result[2]
                    duration = time.time() - t0


                    # print debugging info
                    print("epoch:%5d, step:%5d, trn_loss: %s" % (epoch, step, trn_loss))
                    


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
