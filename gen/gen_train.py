# -*- coding: UTF-8 -*-
import math
import os
import sys

import tensorflow as tf
import numpy as np
import time
from six.moves import xrange
from .gen_model import GenModel
from .gen_data import get_dataset, get_batch
from utils.utils import just_message as just

class GenTrain(object):
    def __init__(self):
        pass

    def pre_train(self, gen_config):
        """
        预训练生成器
        :param gen_config:
        :return:
        """
        print(just("Begin training"))
        with tf.Session() as sess:
            # ① 创建模型
            model = self._create_model(sess, gen_config, forward_only=False, name_scope=gen_config.name_model)

            # ② 获取数据集
            self.train_set, self.train_buckets_scale = self._get_dataset(gen_config)

            # [Ignore]... log相关
            step_time, loss = 0.0, 0.0
            current_step = 0
            gen_loss_summary = tf.Summary()
            gen_writer = tf.summary.FileWriter(gen_config.tensorboard_dir, sess.graph)

            while current_step <= gen_config.max_pre_train_step:
                start_time = time.time() # [Ignore]... log相关：开始时间

                # ③ 获取一个batch的训练数据
                bucket_id = self._get_random_bid()
                encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = get_batch(
                   model, self.train_set, bucket_id, gen_config.batch_size)

                # ④ 获取处理后的输入数据
                # ⑤ 选择训练OP，进行训练
                _, step_loss, _ = self.step(model, sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                             forward_only=False)

                # ================================ [Ignore]... log相关: 记录日志、保存变量 ================================ #

                # log相关：运行时间
                step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
                loss += step_loss / gen_config.steps_per_checkpoint
                current_step += 1

                # 每运行 config_disc.steps_per_checkpoint 次记录一下
                if current_step % gen_config.steps_per_checkpoint == 0:
                    # log相关
                    bucket_value = gen_loss_summary.value.add()
                    bucket_value.tag = gen_config.name_loss
                    bucket_value.simple_value = float(loss)
                    gen_writer.add_summary(gen_loss_summary, int(model.global_step.eval()))

                    # Print statistics for the previous epoch.
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    #     sess.run(model.learning_rate_decay_op)
                    # previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.

                    if current_step % (gen_config.steps_per_checkpoint * 3) == 0:
                        print("current_step: %d, save model" % (current_step))
                        gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
                        if not os.path.exists(gen_ckpt_dir):
                            os.makedirs(gen_ckpt_dir)
                        checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    step_time, loss = 0.0, 0.0
                    # Run evals on development set and print their perplexity.
                    # for bucket_id in xrange(len(gen_config.buckets)):
                    #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    #       dev_set, bucket_id)
                    #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                    #                                target_weights, bucket_id, True)
                    #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                    sys.stdout.flush()

    def step(self, model, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only=True, reward=1, mc_search=False, up_reward=False):
        """
        使用一批数据训练一次模型
        :param model: GenModel
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
        encoder_size, decoder_size = model.buckets[bucket_id]

        # 检查输入数据的长度
        self._check_length(encoder_inputs, decoder_inputs, target_weights, encoder_size, decoder_size)

        # ① 获取处理后的输入数据
        input_feed = self._get_input_feed(model, forward_only, up_reward, mc_search, reward, encoder_inputs,
                        decoder_inputs, target_weights, encoder_size, decoder_size)

        # ② 选择训练OP：取决于是否需要后向传播（如果是测试就只需要预测，不需要更新；如果是训练则需要更新）
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

    def _get_input_feed(self, model, forward_only, up_reward, mc_search, reward, encoder_inputs,
                        decoder_inputs, target_weights, encoder_size, decoder_size):
        """
        给模型中的输入占位符赋予具体的值，这些值源自于get_batch函数的返回值
        Input feed: encoder inputs, decoder inputs, target_weights
        :return: input_feed
        """
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

    def _create_model(self, session, gen_config, forward_only, name_scope, initializer=None):
        """
        创建生成模型：如果已经有训练好的，读入；否则，初始化
        """
        print(just("Creating Gen model: %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim)))
        with tf.variable_scope(name_or_scope=name_scope, initializer=initializer):
            model = GenModel(gen_config, name_scope=name_scope, forward_only=forward_only)
            gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.train_dir, "checkpoints"))
            ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print(just("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path))
                model.saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print(just("Created Gen model with fresh parameters."))
                gen_global_variables = [gv for gv in tf.global_variables() if name_scope in gv.name]
                session.run(tf.variables_initializer(gen_global_variables))
            return model

    def _get_dataset(self, gen_config):
        print(just("Prepare_data"))

        vocab, rev_vocab, dev_set, train_set = get_dataset(gen_config)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]  # 每个桶及前面的桶内数据占总数据的范围

        return train_set, train_buckets_scale

    def _get_random_bid(self):
        """
        随机获取一个训练用的桶id
        :return:
        """
        random_number_01 = np.random.random_sample()
        bucket_id = min(
            [i for i in xrange(len(self.train_buckets_scale)) if self.train_buckets_scale[i] > random_number_01])
        return bucket_id
