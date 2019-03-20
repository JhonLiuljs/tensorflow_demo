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
        """
        1、这个操作的输入logits是未经缩放的，该操作内部会对logits使用softmax操作。
        2、 参数labels,logits必须有相同的形状 [batch_size, num_classes] 和相同的类型。
        3、求交叉熵的公式中常常使用的是以2为底的log函数，这一点便于我们验证
        """
        result1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        result2 = -tf.reduce_mean(self.labels * tf.log(self.logits_scaled), 0)  # 0表示列上行动，加和或者求平均值
        b = self.labels * tf.log(self.logits_scaled)
        a = tf.log(self.logits_scaled)  # 这个表示以2为底的log()
        result3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits_scaled)
        with tf.Session() as sess:
            print("a:", sess.run(a))
            print("b:", sess.run(b))
            print("logits_scaled:", sess.run(self.logits_scaled))
            print("result1:", sess.run(result1))
            print("result2:", sess.run(result2))
            print("result3:", sess.run(result3))

    def sparse_softmax_cross_entropy_with_logits(self, ):
        """
        这是一个TensorFlow中经常需要用到的函数。
        官方文档里面有对它详细的说明，
        传入的logits为神经网络输出层的输出，
        shape为[batch_size，num_classes]，
        传入的label为一个一维的vector，长度等于batch_size，
        每一个值的取值区间必须是[0，num_classes)，其实每一个值就是代表了batch中对应样本的类别。
        这个函数内部会进行两步：
        1、对结果进行softmax归一化
        由于tf.nn.sparse_softmax_cross_entropy_with_logits()输入的label格式为一维的向量，
        所以首先需要将其转化为one-hot格式的编码，
        例如如果分量为3，代表该样本属于第四类，其对应的one-hot格式label为[0，0，0，1，…0]，
        而如果你的label已经是one-hot格式，
        则可以使用tf.nn.softmax_cross_entropy_with_logits()（上面刚刚介绍的）函数来进行softmax和loss的计算。
        :return:
        """
        self.labels = [2, 1]
        logits_scaled = tf.nn.softmax(self.logits)
        result1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        with tf.Session() as sess:
            print("logits_scaled:", sess.run(logits_scaled))
            print("result1:", sess.run(result1))


if __name__ == "__main__":
    loss = Loss()
    loss.softmax_cross_entropy_with_logits()
