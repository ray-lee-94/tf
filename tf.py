import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]='4'
# hello=tf.constant('hello,Tensorflow!')
# sess = tf.Session()
# print(sess.run(hello))
#
# a=tf.constant(2)
# b=tf.constant(3)
# with tf.Session() as sess:
#     print('%i'%sess.run(a))
#
#
# a=tf.placeholder(tf.int16)
# b=tf.placeholder(tf.int16)
#
# add=tf.add(a,b)
# with tf.Session() as sess:
#     print('%i'%sess.run(add,feed_dict={a:1,b:2}))


from tensorflow.examples.tutorials.mnist import input_data
#
mnist=input_data.read_data_sets("/tmp/data",one_hot=False)
#
# Xtr,Ytr=mnist.train.next_batch(5000)
# Xte,Yte=mnist.test.next_batch(200)
#
# xtr=tf.placeholder("float",[None,784])
# xte=tf.placeholder("float",[784])
#
# distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),reduction_indices=1)
# pred=tf.argmin(distance,0)
# accuracy=0
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(len(Xte)):
#         nn_index=sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
#         print(" test",i,"prediction",np.argmax(Ytr[nn_index]),"True class",np.argmax(Yte[i]))
#         if np.argmax(Ytr[nn_index])== np.argmax(Yte[i]):
#             accuracy+=1./len(Xte)
#     print("done")
#     print("acc",accuracy)

import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf
# tf.enable_eager_execution()
# tfe=tf.contrib.eager
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# rng=np.random
lr=0.01
train_epochs=50
display_step=1
num_steps=1500
batch_size=128
# train_x=np.asanyarray([ x*rng.randn()+rng.randn() for x in range(100)])
# train_y=np.asanyarray([x for x in range(100)])
# n_samples=train_x.shape[0]
# x=tf.placeholder("float")
# y=tf.placeholder("float")
# w=tf.variable(rng.randn(),name="weight")
# b=tf.variable(rng.randn(),name="bias")
#
# pred=tf.add(tf.multiply(x,w),b)
# cost=tf.reduce_sum(tf.pow(pred-y,2))/(2*n_samples )
# optimizer=tf.train.gradientdescentoptimizer(lr).minimize(cost)
# init=tf.global_variables_initializer()
#
# with tf.session() as sess:
#     sess.run(init)
#     for epoch in range(train_epochs):
#         for (x,y) in zip(train_x,train_y):
#
#             sess.run(optimizer,feed_dict={x:x,y:y})
#
#         if (epoch+1) %display_step==0:
#             c=sess.run(cost,feed_dict={x:train_x,y:train_y})
#             print("epoch","%04d"%(epoch+1),"cost=","{:.9f}".format(c),"w=",sess.run(w),"b=",sess.run(b))
#     print("finished ")
#     train_cost=sess.run(cost,feed_dict={x:train_x,y:train_y})
#     print("train cost",train_cost)
#     plt.plot(train_x,train_y,'ro',label='original data')
#     plt.plot(train_x,sess.run(w)*train_X+sess.run(b),label='Fitted line')
#     plt.legend()
#     plt.show()
# x=tf.placeholder(tf.float32,[None,784])
# y=tf.placeholder(tf.float32,[None,10])
# W=tf.Variable(tf.zeros([784,10]))
# b=tf.Variable(tf.zeros([10]))
# pred=tf.nn.softmax(tf.matmul(x,W)+b)
# cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
# optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)
# init=tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(train_epochs):
#         avg_cost=0
#         total_batch=int(mnist.train.num_examples/batch_size)
#         for i in range(total_batch):
#             batch_xs,batch_ys=mnist.train.next_batch(batch_size)
#             _,c = sess.run([optimizer,cost],feed_dict={x:batch_xs,y:batch_ys})
#             avg_cost+=c/total_batch
#         if (epoch+1)%display_step==0:
#             print("epoch","%4d"%(epoch+1),"cost=","{:.9f}".format(avg_cost))
#     print("finished!")
#
#     correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
#     accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#     print("accuracy ",accuracy.eval({x:mnist.test.images[:3000],y:mnist.test.labels[:3000]}))
# import numpy as np
# import tensorflow as tf
# from tensorflow.contrib.factorization import KMeans
#
# batch_size=1024
# k=25
# num_classes=10
# num_features=784
# X=tf.placeholder(tf.float32,[None,num_features])
# Y=tf.placeholder(tf.float32,[None,num_classes])
# kmean=KMeans(inputs=X,num_clusters=k,distance_metric="cosine",use_mini_batch=True
#
#              )
# (all_scores,cluster_idx,score,cluster_centers_initialized,cluster_centers_vars,init_op,train_op)=kmean.training_graph()
# cluster_idx=cluster_idx[0]
# avg_distance=tf.reduce_mean(score)
# init_vars=tf.global_variables_initializer()
# sess=tf.Session()
# full_data_x=mnist.train.images
# sess.run(init_vars,feed_dict={X:full_data_x})
# sess.run(init_op,feed_dict={X:full_data_x})
# for i in range(1,100):
#     _,d,idx=sess.run([train_op,avg_distance,cluster_idx],feed_dict={X:full_data_x})
#
# counts=np.zeros([k,num_classes])
# for i in range(len(idx)):
#     counts[idx[i]]+=mnist.train.labels[i]
# labels_map=[np.argmax(c)for c in counts]
# labels_map=tf.convert_to_tensor(labels_map)

n_hidden_1=256
n_hidden_2=256
num_input=784
num_classes=10

# X=tf.placeholder("float",[None,num_input])

# Y=tf.placeholder("float",[None,num_classes])
#
# weights={
#     'h1':tf.Variable(tf.random_normal([num_input,n_hidden_1])),
#     'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
#     'out':tf.Variable(tf.random_normal([n_hidden_2,num_classes]))
#
# }
# biases={
#     'b1':tf.Variable(tf.random_normal([n_hidden_1])) ,
#     'b2':tf.Variable(tf.random_normal([n_hidden_2])) ,
#     'b3':tf.Variable(tf.random_normal([num_classes]))
# }
#
#
# def net(x):
#     layer1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
#     layer2=tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
#     output=tf.matmul(layer2,weights['out'])+biases['b3']
#     return output
# logist=net(X)
#
#
# loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist,labels=Y))
# optimizer=tf.train.AdamOptimizer(lr)
# train_op=optimizer.minimize(loss_op)
# correct_pred=tf.equal(tf.argmax(logist,1),tf.argmax(Y,1))
# accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#
#
# init=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(1,num_steps):
#         batch_x,batch_y=mnist.train.next_batch(batch_size)
#         sess.run(train_op,feed_dict={X:batch_x,Y:batch_y})
#         if epoch%display_step==0 or epoch==1:
#             loss,acc=sess.run([loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y})
#             print("step",str(epoch),"loss","{:.4f}".format(loss),"acc","{:.3f}".format(acc))
#     print("finished")
#     print("test acc ",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
input_fn=tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images},y=mnist.train.labels,batch_size=batch_size,
                                            num_epochs=None,shuffle=True
                                            )

def net(x_dict):
    x=x_dict['images']
    layer1=tf.layers.dense(x,n_hidden_1)
    layer2=tf.layers.dense(layer1,n_hidden_2)
    output=tf.layers.dense(layer2,num_classes)
    return output

def model_fn(features,labels,mode):
    logits=net(features)

    pred_probas=tf.nn.softmax(logits)
    pred_classes=tf.argmax(pred_probas,1)
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,pred_classes)
    loss_op=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.cast(labels,dtype=tf.int32)))
    optimizer=tf.train.GradientDescentOptimizer(lr)
    train_op=optimizer.minimize(loss_op,global_step=tf.train.get_global_step())

    acc_op=tf.metrics.accuracy(labels=labels,predictions=pred_classes)

    estim_specs=tf.estimator.EstimatorSpec(mode=mode,predictions=pred_classes,loss=loss_op,train_op=train_op,
                                           eval_metric_ops={'accuracy':acc_op})
    return estim_specs
model=tf.estimator.Estimator(model_fn)

model.train(input_fn,steps=num_steps)

input_fn=tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images},y=mnist.test.labels,batch_size=batch_size,shuffle=True)
model.evaluate(input_fn)

n_images=4

test_images=mnist.test.images[:n_images]
input_fn=tf.estimator.inputs.numpy_input_fn(x={'images':test_images},shuffle=True)

preds=list(model.predict(input_fn))

for i in range(n_images):
    plt.imshow(np.reshape(test_images[i],[28,28]),cmap='gray')
    plt.show()
    print(preds[i])
