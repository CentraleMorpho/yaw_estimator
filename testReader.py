import tensorflow as tf

if __name__ == '__main__':
	filename_queue2 = tf.train.string_input_producer(['../Cropped/024_poseonly_normalised180.txt'])
	filename_queue = tf.train.string_input_producer(['../Cropped/GeneratedImgs/020/B33_black/P02/857/033000857_003_000002_r073_06.png'])


	reader = tf.WholeFileReader()
	_, value2 = reader.read(filename_queue2)
	key, value = reader.read(filename_queue)



	init_op = tf.initialize_all_variables()
	with tf.Session() as sess:
	  sess.run(init_op)

	  # Start populating the filename queue.

	  coord = tf.train.Coordinator()
	  threads = tf.train.start_queue_runners(coord=coord)

	  for i in range(1): 
	    val2 = value2.eval() 
	    key = key.eval()

	 
	  #print(val2)
	  print(key)

	  coord.request_stop()
	  coord.join(threads)
