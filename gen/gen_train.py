# -*- coding: UTF-8 -*-

import numpy as np
from six.moves import xrange

class GenTrain(object):
    def __init__(self, model):
        self.model = model # GenModel

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only=True, reward=1, mc_search=False, up_reward=False):
        """
        使用一批数据训练一次模型
        :param session:
        :param encoder_inputs: 一个list，time-major问题数据，由id表示
        :param decoder_inputs: 一个list，time-major回答数据，由id表示
        :param target_weights: 一个list，time-major权值数据，由1和0表示
        :param bucket_id:
        :param forward_only:
        :param reward:
        :param mc_search:
        :param up_reward:
        :return:
        """
        encoder_size, decoder_size = self.model.buckets[bucket_id]

        model = self.model

        # 检查输入数据的长度
        self._check_length(encoder_inputs, decoder_inputs, target_weights, encoder_size, decoder_size)

        # ① 获取输入数据
        input_feed = self._get_input_feed(forward_only, up_reward, mc_search, reward, encoder_inputs,
                        decoder_inputs, target_weights, encoder_size, decoder_size)

        # ② 设置训练OP：取决于是否需要后向传播（如果是测试就只需要预测，不需要更新；如果是训练则需要更新）
        if not forward_only:  # 训练
            output_feed = [model.updates[bucket_id],  # Update Op that does SGD.
                           model.aj_losses[bucket_id],  # Gradient norm.
                           model.losses[bucket_id]]  # Loss for this batch.
        else:  # 测试或强化学习 TODO(Zhu) 为什么强化学习不需要后向传播？
            output_feed = [model.encoder_state[bucket_id], model.losses[bucket_id]]  # Loss for this batch.
            for l in xrange(decoder_size):
                # model.outputs[bucket_id][l]的形状：[batch_size, target_vocab_size]，表示每个时间步的单词预测概率
                # 相关代码见 GenModel._rl_seq2seq_model()
                output_feed.append(model.outputs[bucket_id][l])

        # ③ 训练
        outputs = session.run(output_feed, input_feed)

        # ④ 输出
        if not forward_only:
            return outputs[1], outputs[2], outputs[0]  # Gradient norm, loss, no outputs.
        else:
            return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.

    def _check_length(self, encoder_inputs, decoder_inputs, target_weights, encoder_size, decoder_size):
        if len(encoder_inputs) != encoder_size: # encoder_inputs是time-major，第一维是句子长度
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

    def _get_input_feed(self, forward_only, up_reward, mc_search, reward, encoder_inputs,
                        decoder_inputs, target_weights, encoder_size, decoder_size):
        """
        给模型中的输入占位符赋予具体的值，这些值源自于get_batch函数的返回值
        Input feed: encoder inputs, decoder inputs, target_weights
        :return: input_feed
        """
        model = self.model
        input_feed = {
            model.forward_only.name: forward_only,
            model.up_reward.name: up_reward,
            model.mc_search.name: mc_search
        }
        for l in xrange(len(model.buckets)):
            input_feed[model.reward[l].name] = reward
        for l in xrange(encoder_size):
            input_feed[model.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[model.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[model.target_weights[l].name] = target_weights[l]

        # 在decoder_inputs的最后一列添加<EOS>（用0表示），decoder_inputs的长度是decoder_size+1
        # 模型中会将decoder_inputs向前移进一个元素得到target，相关代码见 GenModel._init_input_placeholders()
        last_target = model.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

        return input_feed