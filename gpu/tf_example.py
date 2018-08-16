# coding:utf-8
import tensorflow as tf

# allow_soft_placement=True 表示使用不能使用显卡时使用cpu
# log_device_placement=False 不打印日志，不然会刷屏

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    a = tf.constant(1)
    b = tf.constant(3)
    c = a + b
    print('结果是：%d' % sess.run(c))

# new  graph.
x = 3
y = 4
z = 2
with tf.device('/gpu:0'):
    a = tf.multiply(x, x)
    b = tf.multiply(a, y)
with tf.device('/gpu:1'):
    c = tf.add(y, z)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
d = tf.add(b, c)

print(sess.run(d))
