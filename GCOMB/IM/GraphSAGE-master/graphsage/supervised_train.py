from __future__ import division
from __future__ import print_function

import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import os
import time
from pathlib import Path
import tensorflow as tf
import numpy as np
import sklearn
from sklearn import metrics

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'graphsage_maxpool', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string("weight_model", "WC", "Can be TV/WC/CONST/LND")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('dataset', 'youtube', 'dataset to train')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 5000, 'number of epochs to train.')
flags.DEFINE_integer('saved_every_n_epochs', 100, 'every n epoch to save')
flags.DEFINE_float('dropout', 0.1, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.00002, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 15, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 25, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 60, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 60, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 100, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8
dataset = FLAGS.dataset
weight_model = FLAGS.weight_model
MODEL_PATH = "%s_supervisedTrainedModel_MC_marginal/model.ckpt" % weight_model


# def calc_f1(y_true, y_pred):
#     if not FLAGS.sigmoid:
#         y_true = np.argmax(y_true, axis=1)
#         y_pred = np.argmax(y_pred, axis=1)
#     else:
#         y_pred[y_pred > 0.5] = 1
#         y_pred[y_pred <= 0.5] = 0
#     return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                             feed_dict=feed_dict_val)
    # mic, mac = calc_f1(labels, node_outs_val[0])
    return node_outs_val[1], (time.time() - t_test)


def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def predict_alldata(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                                                 test=True)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1

    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    # f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), (time.time() - t_test)


def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _ = minibatch_iter.incremental_node_val_feed_dict(size, iter_num,
                                                                                                 test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                                 feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    # f1_scores = calc_f1(labels, val_preds)
    return np.mean(val_losses), (time.time() - t_test)


def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders


def train(train_data, test_data=None):
    G = train_data[0]
    # features = train_data[1]
    id_map = train_data[1]
    class_map = train_data[3]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
    #
    # if not features is None:
    #     # pad with dummy zero vector
    #     features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch = NodeMinibatchIterator(G,
                                      id_map,
                                      placeholders,
                                      class_map,
                                      num_classes,
                                      batch_size=FLAGS.batch_size,
                                      max_degree=FLAGS.max_degree,
                                      context_pairs=context_pairs, mode="train", prefix=FLAGS.train_prefix)

    features = minibatch.features

    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        if FLAGS.samples_3 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                           SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
        elif FLAGS.samples_2 != 0:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                           SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
        else:
            layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos,
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="gcn",
                                    model_size=FLAGS.model_size,
                                    concat=False,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="seq",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="maxpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SupervisedGraphsage(num_classes, placeholders,
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos,
                                    aggregator_type="meanpool",
                                    model_size=FLAGS.model_size,
                                    sigmoid_loss=FLAGS.sigmoid,
                                    identity_dim=FLAGS.identity_dim,
                                    logging=True)

    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    # Train model
    print(layer_infos)
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    # Load trained model
    if Path(MODEL_PATH).parent.exists():
        print("Entered in the Loop  =           == = == = == ")
        var_to_save = []
        for var in tf.trainable_variables():
            var_to_save.append(var)
        saver = tf.train.Saver(var_to_save, max_to_keep=1000)
        print("Trained Model Loading!")
        saver.restore(sess, tf.train.latest_checkpoint(Path(MODEL_PATH).parent))
        print("Trained Model Loaded!")

    # ------------------------------------------------------------------

    for epoch in range(1, FLAGS.epochs + 1):
        minibatch.shuffle()

        iter = 0
        print('Epoch: %04d' % epoch)
        epoch_val_costs.append(0)
        while not minibatch.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            #
            # if iter % FLAGS.validate_iter == 0:
            #     # Validation
            #     sess.run(val_adj_info.op)
            #     if FLAGS.validate_batch_size == -1:
            #         val_cost, duration = incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
            #     else:
            #         val_cost, duration = evaluate(sess, model, minibatch, FLAGS.validate_batch_size)
            #     sess.run(train_adj_info.op)
            #     epoch_val_costs[-1] += val_cost
            #
            # if total_steps % FLAGS.print_every == 0:
            #     summary_writer.add_summary(outs[0], total_steps)

            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                # train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter,
                      "train_loss=", "{:.8f}".format(train_cost),
                      #   "val_loss=", "{:.8f}".format(val_cost),
                      "time=", "{:.5f}".format(avg_time))

            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
            break

    print("Optimization Finished!")
    var_to_save = []
    for var in tf.trainable_variables():
        var_to_save.append(var)
    saver = tf.train.Saver(var_to_save)
    save_path = saver.save(sess, MODEL_PATH, write_meta_graph=False)
    print("*** Saved: Model", save_path + '-' + str(epoch))

    print("Writing test set stats to file (don't peak!)")


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    print("Done loading training data..")
    start = time.time()
    train(train_data)
    print("Training GraphSAGE takes: ", time.time() - start)


if __name__ == '__main__':
    tf.app.run()
