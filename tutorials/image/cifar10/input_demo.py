import tensorflow as tf


with tf.Session() as sess:
    filename = ['A.jpg','B.jpg']
    #string_input_produce接收一个文件列表作为输入，将文件读入文件名队列中，以供内存队列调用，shuffle表示是否打乱，num_epochs表示一个文件读几次
    #要注意的是创建文件列表的时候并没有开始填充队列，必须执行tf.train.start_queue_runners
    filename_queue = tf.train.string_input_producer(filename,shuffle=True,num_epochs=5)
    #WholeFileReader是一个内存队列的对象，他的read()方法接收一个文件名队列
    reader = tf.WholeFileReader()
    key,value = reader.read(filename_queue)
    #必须要初始化所有变量，才能分配内存
    tf.local_variables_initializer().run()
    #start_queue_runners后，才会开始填充文件名队列
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        #sess.run表示获取一次值
        image_data = sess.run(value)

