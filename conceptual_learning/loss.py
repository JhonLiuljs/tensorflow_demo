# coding:utf-8
"""
loss测试
"""
import tensorflow as tf


class Loss:
    def __init__(self, data_type=1):
        self.labels = [[0.2, 0.3, 0.5],
                       [0.1, 0.6, 0.3]]
        self.logits = [[2, 0.5, 1],
                       [0.1, 1, 3]]
        self.logits_scaled = tf.nn.softmax(self.logits)

    def softmax_cross_entropy_with_logits(self, ):
        result1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        result2 = -tf.reduce_mean(self.labels*tf.log(self.logits_scaled),0)### 0表示列上行动，加和或者求平均值
        b=self.labels*tf.log(self.logits_scaled)
        a=tf.log(self.logits_scaled)###这个表示以2为底的log()
        result3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_scaled)
        with tf.Session() as sess:
            print ("a:",sess.run(a))
            print ("b:",sess.run(b))
            print ("logits_scaled:",sess.run(self.logits_scaled))
            print ("result1:",sess.run(result1))
            print ("result2:",sess.run(result2))
            print ("result3:",sess.run(result3))


if __name__ == "__main__":
    loss = Loss()
    loss.softmax_cross_entropy_with_logits()
