# -*- coding: UTF-8 -*-


def get_batch(self, train_data, bucket_id, batch_size, type=0):
    """Get a random batch of data from the specified bucket, prepare for step.
    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.
    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """

    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data, 用到的type都是0，都是从桶里随机挑一组
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_source_encoder, batch_source_decoder = [], []

    if type == 1:
        batch_size = 1
    for batch_i in xrange(batch_size):
        if type == 1:  # 返回桶内所有数据
            encoder_input, decoder_input = train_data[bucket_id]
        elif type == 2:  # 取桶内第一组，encoder_input是第batch_i个单词，encoder只有一个单词 # TODO 但下面是把它当数组用的，这里就报错了
            # print("disc_data[bucket_id]: ", disc_data[bucket_id][0])
            encoder_input_a, decoder_input = train_data[bucket_id][0]
            encoder_input = encoder_input_a[batch_i]
        elif type == 0:  # 桶内随机挑一组
            encoder_input, decoder_input = random.choice(train_data[bucket_id])
            # print("train en: %s, de: %s" %(encoder_input, decoder_input))

        batch_source_encoder.append(encoder_input)
        batch_source_decoder.append(decoder_input)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the disc_data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in xrange(batch_size)], dtype=np.int32))

        # Create target_weights to be 0 for targets that are padding.
        batch_weight = np.ones(batch_size, dtype=np.float32)
        for batch_idx in xrange(batch_size):
            # We set weight to 0 if the corresponding target is a PAD symbol. 如果是PAD，设置它的权值为0
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder)