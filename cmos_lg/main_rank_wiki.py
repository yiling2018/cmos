import tensorflow as tf
import numpy as np
import math
import time
import os
import input_rank_wiki as my_input
import metrics

image1_size = my_input.alog_size
image2_size = my_input.log_size
image3_size = my_input.pred_size
image4_size = my_input.m5e_size
image5_size = my_input.m6h_size
image6_size = my_input.m7d_size
text_size = my_input.text_size

batch_size = 30
max_steps = 10000
lamda = 0.00001
gamma = 0.001
log_dir = 'log'
is_train = True # training or testing
is_cont = False  # continue training


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def run_training():
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        image1_inputs = tf.placeholder(tf.float32, shape=[None, image1_size])
        image2_inputs = tf.placeholder(tf.float32, shape=[None, image2_size])
        image3_inputs = tf.placeholder(tf.float32, shape=[None, image3_size])
        image4_inputs = tf.placeholder(tf.float32, shape=[None, image4_size])
        image5_inputs = tf.placeholder(tf.float32, shape=[None, image5_size])
        image6_inputs = tf.placeholder(tf.float32, shape=[None, image6_size])
        text_inputs = tf.placeholder(tf.float32, shape=[None, text_size])

        image1_1, image1_2, image1_3 = tf.split(image1_inputs, 3, 0)
        image2_1, image2_2, image2_3 = tf.split(image2_inputs, 3, 0)
        image3_1, image3_2, image3_3 = tf.split(image3_inputs, 3, 0)
        image4_1, image4_2, image4_3 = tf.split(image4_inputs, 3, 0)
        image5_1, image5_2, image5_3 = tf.split(image5_inputs, 3, 0)
        image6_1, image6_2, image6_3 = tf.split(image6_inputs, 3, 0)
        t_1, t_2, t_3 = tf.split(text_inputs, 3, 0)

        with tf.name_scope('image1'):
            s_w1 = tf.Variable(
                tf.truncated_normal([image1_size, text_size], stddev=1.0 / math.sqrt(float(image1_size))))
            tf.add_to_collection('weight_decay_w', s_w1)
            image1_sp_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image1_2, s_w1), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image1_sn_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image1_3, s_w1), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image1_sp_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image1_1, s_w1), axis=1), tf.expand_dims(t_2, axis=2)), axis=1)
            image1_sn_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image1_1, s_w1), axis=1), tf.expand_dims(t_3, axis=2)), axis=1)
            g_w1 = tf.Variable(
                tf.truncated_normal([image1_size, 1], stddev=1.0 / math.sqrt(float(image1_size))))
            tf.add_to_collection('weight_decay', g_w1)
            image1_gate = tf.matmul(image1_inputs, g_w1)

        with tf.name_scope('image2'):
            s_w2 = tf.Variable(
                tf.truncated_normal([image2_size, text_size], stddev=1.0 / math.sqrt(float(image2_size))))
            tf.add_to_collection('weight_decay_w', s_w2)
            image2_sp_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image2_2, s_w2), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image2_sn_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image2_3, s_w2), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image2_sp_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image2_1, s_w2), axis=1), tf.expand_dims(t_2, axis=2)), axis=1)
            image2_sn_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image2_1, s_w2), axis=1), tf.expand_dims(t_3, axis=2)), axis=1)
            g_w2 = tf.Variable(
                tf.truncated_normal([image2_size, 1], stddev=1.0 / math.sqrt(float(image2_size))))
            tf.add_to_collection('weight_decay', g_w2)
            image2_gate = tf.matmul(image2_inputs, g_w2)

        with tf.name_scope('image3'):
            s_w3 = tf.Variable(
                tf.truncated_normal([image3_size, text_size], stddev=1.0 / math.sqrt(float(image3_size))))
            tf.add_to_collection('weight_decay_w', s_w3)
            image3_sp_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image3_2, s_w3), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image3_sn_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image3_3, s_w3), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image3_sp_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image3_1, s_w3), axis=1), tf.expand_dims(t_2, axis=2)), axis=1)
            image3_sn_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image3_1, s_w3), axis=1), tf.expand_dims(t_3, axis=2)), axis=1)
            g_w3 = tf.Variable(
                tf.truncated_normal([image3_size, 1], stddev=1.0 / math.sqrt(float(image3_size))))
            tf.add_to_collection('weight_decay', g_w3)
            image3_gate = tf.matmul(image3_inputs, g_w3)

        with tf.name_scope('image4'):
            s_w4 = tf.Variable(
                tf.truncated_normal([image4_size, text_size], stddev=1.0 / math.sqrt(float(image4_size))))
            tf.add_to_collection('weight_decay_w', s_w4)
            image4_sp_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image4_2, s_w4), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image4_sn_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image4_3, s_w4), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image4_sp_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image4_1, s_w4), axis=1), tf.expand_dims(t_2, axis=2)), axis=1)
            image4_sn_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image4_1, s_w4), axis=1), tf.expand_dims(t_3, axis=2)), axis=1)
            g_w4 = tf.Variable(
                tf.truncated_normal([image4_size, 1], stddev=1.0 / math.sqrt(float(image4_size))))
            tf.add_to_collection('weight_decay', g_w4)
            image4_gate = tf.matmul(image4_inputs, g_w4)

        with tf.name_scope('image5'):
            s_w5 = tf.Variable(
                tf.truncated_normal([image5_size, text_size], stddev=1.0 / math.sqrt(float(image5_size))))
            tf.add_to_collection('weight_decay_w', s_w5)
            image5_sp_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image5_2, s_w5), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image5_sn_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image5_3, s_w5), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image5_sp_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image5_1, s_w5), axis=1), tf.expand_dims(t_2, axis=2)), axis=1)
            image5_sn_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image5_1, s_w5), axis=1), tf.expand_dims(t_3, axis=2)), axis=1)
            g_w5 = tf.Variable(
                tf.truncated_normal([image5_size, 1], stddev=1.0 / math.sqrt(float(image5_size))))
            tf.add_to_collection('weight_decay', g_w5)
            image5_gate = tf.matmul(image5_inputs, g_w5)

        with tf.name_scope('image6'):
            s_w6 = tf.Variable(
                tf.truncated_normal([image6_size, text_size], stddev=1.0 / math.sqrt(float(image6_size))))
            tf.add_to_collection('weight_decay_w', s_w6)
            image6_sp_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image6_2, s_w6), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image6_sn_it = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image6_3, s_w6), axis=1), tf.expand_dims(t_1, axis=2)), axis=1)
            image6_sp_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image6_1, s_w6), axis=1), tf.expand_dims(t_2, axis=2)), axis=1)
            image6_sn_ti = tf.squeeze(tf.matmul(tf.expand_dims(tf.matmul(image6_1, s_w6), axis=1), tf.expand_dims(t_3, axis=2)), axis=1)
            g_w6 = tf.Variable(
                tf.truncated_normal([image6_size, 1], stddev=1.0 / math.sqrt(float(image6_size))))
            tf.add_to_collection('weight_decay', g_w6)
            image6_gate = tf.matmul(image6_inputs, g_w6)

        gate = tf.nn.softmax(tf.stack([image1_gate, image2_gate, image3_gate, image4_gate, image5_gate, image6_gate], axis=1), dim=1)
        gate_1, gate_2, gate_3 = tf.split(gate, 3, 0)
        sp_it = tf.reduce_sum(tf.multiply(tf.stack([image1_sp_it, image2_sp_it, image3_sp_it, image4_sp_it, image5_sp_it, image6_sp_it], axis=1), gate_2), axis=1)
        sn_it = tf.reduce_sum(tf.multiply(tf.stack([image1_sn_it, image2_sn_it, image3_sn_it, image4_sn_it, image5_sn_it, image6_sn_it], axis=1), gate_3), axis=1)
        sp_ti = tf.reduce_sum(tf.multiply(tf.stack([image1_sp_ti, image2_sp_ti, image3_sp_ti, image4_sp_ti, image5_sp_ti, image6_sp_ti], axis=1), gate_1), axis=1)
        sn_ti = tf.reduce_sum(tf.multiply(tf.stack([image1_sn_ti, image2_sn_ti, image3_sn_ti, image4_sn_ti, image5_sn_ti, image6_sn_ti], axis=1), gate_1), axis=1)

        loss_it = tf.reduce_mean(tf.maximum(0., 1 + tf.subtract(sn_it, sp_it)))
        loss_ti = tf.reduce_mean(tf.maximum(0., 1 + tf.subtract(sn_ti, sp_ti)))

        vars_w = tf.get_collection('weight_decay_w')
        vars = tf.get_collection('weight_decay')
        loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in vars_w]) * lamda + tf.add_n([tf.nn.l2_loss(v) for v in vars]) * gamma
        loss = 1*loss_it + 1*loss_ti + loss_l2

        # Construct the ADAM optimizer
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Add variable initializer.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

    train_data_sets = my_input.get_train_data()
    test_data_sets = my_input.get_test_data()

    # Begin training.
    with tf.Session(graph=graph) as sess:

        # We must initialize all variables before we use them.
        sess.run(init)
        print("Initialized")

        average_loss = 0
        start_time = time.time()
        for step in range(max_steps):
            batch_image1, batch_image2, batch_image3, batch_image4, batch_image5, batch_image6, batch_text = train_data_sets.next_batch(batch_size)
            feed_dict = {image1_inputs: batch_image1, image2_inputs: batch_image2, image3_inputs: batch_image3,
                         image4_inputs: batch_image4, image5_inputs: batch_image5, image6_inputs: batch_image6, text_inputs: batch_text}

            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            duration = time.time() - start_time
            # Write the summaries and print an overview fairly often.
            if (step + 1) % 500 == 0:
                average_loss /= 500
                print('Step %d: average_loss = %.2f (%.3f sec)' % (step, average_loss, duration))
                average_loss = 0

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 2000 == 0 or (step + 1) == max_steps:
                save_path = saver.save(sess, os.path.join(log_dir, "model"), step)
                ("Model saved in file: %s" % save_path)

                batch_image1, batch_image2, batch_image3, batch_image4, batch_image5, batch_image6, batch_text, batch_label = test_data_sets.next_batch()
                feed_dict = {image1_inputs: batch_image1, image2_inputs: batch_image2, image3_inputs: batch_image3,
                             image4_inputs: batch_image4, image5_inputs: batch_image5, image6_inputs: batch_image6,
                             text_inputs: batch_text}
                [gw1, gw2, gw3, gw4, gw5, gw6, sw1, sw2, sw3, sw4, sw5, sw6] = sess.run(
                    [g_w1, g_w2, g_w3, g_w4, g_w5, g_w6, s_w1, s_w2, s_w3, s_w4, s_w5, s_w6], feed_dict=feed_dict)
                temp = np.concatenate(
                    [np.dot(batch_image1, gw1), np.dot(batch_image2, gw2), np.dot(batch_image3, gw3),
                     np.dot(batch_image4, gw4), np.dot(batch_image5, gw5), np.dot(batch_image6, gw6)], axis=1)
                temp = softmax(temp)
                scores = temp[:, [0]] * np.dot(np.dot(batch_image1, sw1), batch_text.T) + \
                         temp[:, [1]] * np.dot(np.dot(batch_image2, sw2), batch_text.T) + \
                         temp[:, [2]] * np.dot(np.dot(batch_image3, sw3), batch_text.T) + \
                         temp[:, [3]] * np.dot(np.dot(batch_image4, sw4), batch_text.T) + \
                         temp[:, [4]] * np.dot(np.dot(batch_image5, sw5), batch_text.T) + \
                         temp[:, [5]] * np.dot(np.dot(batch_image6, sw6), batch_text.T)
                map_i2t, map_t2i = metrics.evaluate_s(scores, batch_label)
                with open('record.txt', 'a') as f:
                    f.write('%d %f %f\n'% (step + 1, map_i2t[-1], map_t2i[-1]))


def run_testing():
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        image1_inputs = tf.placeholder(tf.float32, shape=[None, image1_size])
        image2_inputs = tf.placeholder(tf.float32, shape=[None, image2_size])
        image3_inputs = tf.placeholder(tf.float32, shape=[None, image3_size])
        image4_inputs = tf.placeholder(tf.float32, shape=[None, image4_size])
        image5_inputs = tf.placeholder(tf.float32, shape=[None, image5_size])
        image6_inputs = tf.placeholder(tf.float32, shape=[None, image6_size])
        text_inputs = tf.placeholder(tf.float32, shape=[None, text_size])

        with tf.name_scope('image1'):
            s_w1 = tf.Variable(
                tf.truncated_normal([image1_size, text_size], stddev=1.0 / math.sqrt(float(image1_size))))
            tf.add_to_collection('weight_decay_w', s_w1)
            g_w1 = tf.Variable(
                tf.truncated_normal([image1_size, 1], stddev=1.0 / math.sqrt(float(image1_size))))
            tf.add_to_collection('weight_decay', g_w1)

        with tf.name_scope('image2'):
            s_w2 = tf.Variable(
                tf.truncated_normal([image2_size, text_size], stddev=1.0 / math.sqrt(float(image2_size))))
            tf.add_to_collection('weight_decay_w', s_w2)
            g_w2 = tf.Variable(
                tf.truncated_normal([image2_size, 1], stddev=1.0 / math.sqrt(float(image2_size))))
            tf.add_to_collection('weight_decay', g_w2)

        with tf.name_scope('image3'):
            s_w3 = tf.Variable(
                tf.truncated_normal([image3_size, text_size], stddev=1.0 / math.sqrt(float(image3_size))))
            tf.add_to_collection('weight_decay_w', s_w3)
            g_w3 = tf.Variable(
                tf.truncated_normal([image3_size, 1], stddev=1.0 / math.sqrt(float(image3_size))))
            tf.add_to_collection('weight_decay', g_w3)

        with tf.name_scope('image4'):
            s_w4 = tf.Variable(
                tf.truncated_normal([image4_size, text_size], stddev=1.0 / math.sqrt(float(image4_size))))
            tf.add_to_collection('weight_decay_w', s_w4)
            g_w4 = tf.Variable(
                tf.truncated_normal([image4_size, 1], stddev=1.0 / math.sqrt(float(image4_size))))
            tf.add_to_collection('weight_decay', g_w4)

        with tf.name_scope('image5'):
            s_w5 = tf.Variable(
                tf.truncated_normal([image5_size, text_size], stddev=1.0 / math.sqrt(float(image5_size))))
            tf.add_to_collection('weight_decay_w', s_w5)
            g_w5 = tf.Variable(
                tf.truncated_normal([image5_size, 1], stddev=1.0 / math.sqrt(float(image5_size))))
            tf.add_to_collection('weight_decay', g_w5)

        with tf.name_scope('image6'):
            s_w6 = tf.Variable(
                tf.truncated_normal([image6_size, text_size], stddev=1.0 / math.sqrt(float(image6_size))))
            tf.add_to_collection('weight_decay_w', s_w6)
            g_w6 = tf.Variable(
                tf.truncated_normal([image6_size, 1], stddev=1.0 / math.sqrt(float(image6_size))))
            tf.add_to_collection('weight_decay', g_w6)

        # Create a saver for loading training checkpoints.
        saver = tf.train.Saver()

    test_data_sets = my_input.get_test_data()

    # Begin training.
    with tf.Session(graph=graph) as sess:
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if not ckpt or not ckpt.model_checkpoint_path:
            print('no checkpoint file.')
            return

        model = ckpt.model_checkpoint_path
        # model = 'log\\model-59999'
        saver.restore(sess, model)
        print("Restored")

        batch_image1, batch_image2, batch_image3, batch_image4, batch_image5, batch_image6, batch_text, batch_label = test_data_sets.next_batch()
        feed_dict = {image1_inputs: batch_image1, image2_inputs: batch_image2, image3_inputs: batch_image3,
                     image4_inputs: batch_image4, image5_inputs: batch_image5, image6_inputs: batch_image6,
                     text_inputs: batch_text}
        [gw1, gw2, gw3, gw4, gw5, gw6, sw1, sw2, sw3, sw4, sw5, sw6] = sess.run(
            [g_w1, g_w2, g_w3, g_w4, g_w5, g_w6, s_w1, s_w2, s_w3, s_w4, s_w5, s_w6], feed_dict=feed_dict)
        temp = np.concatenate(
            [np.dot(batch_image1, gw1), np.dot(batch_image2, gw2), np.dot(batch_image3, gw3), np.dot(batch_image4, gw4),
             np.dot(batch_image5, gw5), np.dot(batch_image6, gw6)], axis=1)
        temp = softmax(temp)
        scores = temp[:, [0]] * np.dot(np.dot(batch_image1, sw1), batch_text.T) + \
                 temp[:, [1]] * np.dot(np.dot(batch_image2, sw2), batch_text.T) + \
                 temp[:, [2]] * np.dot(np.dot(batch_image3, sw3), batch_text.T) + \
                 temp[:, [3]] * np.dot(np.dot(batch_image4, sw4), batch_text.T) + \
                 temp[:, [4]] * np.dot(np.dot(batch_image5, sw5), batch_text.T) + \
                 temp[:, [5]] * np.dot(np.dot(batch_image6, sw6), batch_text.T)
        metrics.evaluate_s(scores, batch_label)


def main(_):
    if is_train:
        if not is_cont:
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)
        run_training()
    else:
        run_testing()

if __name__ == "__main__":
    tf.app.run()
