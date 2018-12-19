#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import statistical_analysis as sa
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("source", "tensorflow docker vscode bootstrap", "source projects, split by space")
tf.flags.DEFINE_string("target", "DECA", "target project")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 192, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularizaion lambda (default: 0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 1000)")
tf.flags.DEFINE_integer("evaluate_every", 1, "Evaluate model on dev set after this many epochs (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


#sentence class
class_names = ('information giving', 'information seeking', 'feature request', 'solution proposal', 'problem discovery')#, 'aspect evaluation', 'others')


def train_model(allow_save_model = False, print_intermediate_results = True):

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Data Preparatopn
    # ==================================================

    # Load data
    print("Loading data...")

    #load each source project
    projects = FLAGS.source.split()
    source_text = np.array([])
    source_y = np.array([])
    for project in projects:
        source_file_path = "data/" + project + "/"
        source_files = list()
        for class_name in class_names:
            source_files.append(source_file_path + class_name)
        tmp_text, tmp_y = data_helpers.load_data_and_labels(source_files)
        print project+": "+str(len(tmp_text))+" sentences"
        source_text = np.concatenate([source_text, tmp_text], 0)
        if len(source_y)==0:
            source_y = np.array(tmp_y)
        else:
            source_y = np.concatenate([source_y, tmp_y], 0)

    #load target project
    # target_file_path = "data/" + FLAGS.target + "/"
    # target_files = list()
    # for class_name in class_names:
    #     target_files.append(target_file_path + class_name)
    # target_text, target_y = data_helpers.load_data_and_labels(target_files)

    # load target project
    projects = FLAGS.target.split()
    target_text = np.array([])
    target_y = np.array([])
    for project in projects:
        target_file_path = "data/" + project + "/"
        target_files = list()
        for class_name in class_names:
            target_files.append(target_file_path + class_name)
        tmp_text, tmp_y = data_helpers.load_data_and_labels(target_files)
        print project + ": " + str(len(tmp_text)) + " sentences"
        target_text = np.concatenate([target_text, tmp_text], 0)
        if len(target_y) == 0:
            target_y = np.array(tmp_y)
        else:
            target_y = np.concatenate([target_y, tmp_y], 0)


    all_text = np.concatenate([source_text, target_text], 0)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in all_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    source_x = np.array(list(vocab_processor.fit_transform(source_text)))
    target_x = np.array(list(vocab_processor.fit_transform(target_text)))

    if print_intermediate_results:
        print('data distribution in source dataset')
        sa.print_data_distribution(source_y, class_names)
        print('data distribution in target dataset')
        sa.print_data_distribution(target_y, class_names)

        print("Max Document Length: {:d}".format(max_document_length))
        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Test size: {:d}/{:d}".format(len(source_y), len(target_y)))

    # Training
    # ==================================================

    min_loss = 100000000
    predictions_at_min_loss = None
    steps_per_epoch = (int)(len(source_y) / FLAGS.batch_size) + 1

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=source_x.shape[1],
                num_classes=source_y.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)

            learning_rate = tf.train.polynomial_decay(2*1e-3, global_step,
                                                      steps_per_epoch * FLAGS.num_epochs, 1e-4,
                                                      power=1)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            if allow_save_model:

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(
                    os.path.join(os.path.curdir, "runs", FLAGS.source))
                print("Writing to {}\n".format(out_dir))

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir_name = "checkpoint-" + str(FLAGS.embedding_dim) + "-" + FLAGS.filter_sizes + "-" + \
                                      str(FLAGS.num_filters) + "-" + str(FLAGS.dropout_keep_prob) + "-" + str(
                    FLAGS.l2_reg_lambda) + \
                                      "-" + str(FLAGS.batch_size) + "-" + str(FLAGS.num_epochs)
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, checkpoint_dir_name))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.all_variables())

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer(), feed_dict={cnn.phase_train: True})  # this is for version r0.12

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.phase_train: True
                }
                _, step, loss, mean_loss, l2_loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.mean_loss, cnn.l2_loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}, mean_loss {}, l2_loss {}".format(time_str, step, loss, accuracy, mean_loss, l2_loss))
                return accuracy

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0,
                    cnn.phase_train: False
                }
                step, loss, mean_loss, l2_loss, accuracy, predictions = sess.run(
                    [global_step, cnn.loss, cnn.mean_loss, cnn.l2_loss, cnn.accuracy, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                if print_intermediate_results:
                    print("{}: epoch {}, step {}, loss {:g}, acc {:g}, mean_loss {}, l2_loss {}".format(
                        time_str, step/steps_per_epoch, step, loss, accuracy, mean_loss, l2_loss))
                return accuracy, loss, predictions

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(source_x, source_y)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_accuracy = train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                current_epoch = current_step/steps_per_epoch
                if current_step%steps_per_epoch==0 and current_epoch % FLAGS.evaluate_every == 0:
                    if print_intermediate_results:
                        print("Current train accuracy: %s\nEvaluation:" % (train_accuracy))

                    fold_accuracy, loss, predictions = dev_step(target_x, target_y)
                    #ensemble_prediction([predictions], target_y)
                    if loss < min_loss:
                        min_loss = loss
                        predictions_at_min_loss = predictions
                        if allow_save_model:
                            save_path = saver.save(sess, checkpoint_prefix)
                            if print_intermediate_results:
                                print("Model saved in file: %s" % save_path)


            # Final result
            output_file = open(target_file_path + 'fp_sentences', 'w')
            print('Final result:')
            fold_accuracy, loss, predictions = dev_step(target_x, target_y)
            print("ACC: %s" % (fold_accuracy))
            tp, fp, fn, precision, recall, f1 = sa.calculate_IR_metrics(target_text, target_y, predictions, class_names,
                                                                        output_file)
            for i in range(len(class_names)):
                print class_names[i], precision[i], recall[i], f1[i]
            print("average f1-score: %s" % (sum(f1) / len(f1)))

            output_file.close()

    return min_loss, predictions_at_min_loss, target_y

def ensemble_prediction (list_predictions, y_classes):

    tp, fp, fn, precision, recall, f1 = (np.zeros(len(class_names)) for _ in range(6))

    num_instances = len(y_classes)
    votes = list()
    for _ in range(num_instances):
        votes.append((np.zeros(len(class_names))))


    for prediction_epoch in list_predictions:
        for instance_index, predicted_class in enumerate(prediction_epoch):
            votes[instance_index][predicted_class] += 1

    final_prediction = np.zeros(num_instances)

    for instance_index,true_class in enumerate(y_classes):
        true_class = np.argmax(true_class)
        predict_class = np.argmax(votes[instance_index])
        final_prediction[instance_index] = predict_class
        if predict_class == true_class:
            tp[true_class] += 1
        if predict_class != true_class:
            fn[true_class] += 1
            fp[predict_class] += 1

    print ("final accuracy:",sum(tp)*1.0/len(y_classes))

    for i in range(len(class_names)):
        precision[i] = tp[i] / (tp[i] + fp[i])
        recall[i] = tp[i] / (tp[i] + fn[i])
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

    for i in range(len(class_names)):
        print class_names[i], 'precision: ', precision[i], ', recall: ', recall[i], ', f1-score: ', f1[i]


def auto_param_tuning():


    tmp = FLAGS.embedding_dim
    best_embedding_dim = FLAGS.embedding_dim
    min_loss = 100000000
    list_embedding_dim = {128, 192, 256, 320}
    for embedding_dim in list_embedding_dim:
        FLAGS.embedding_dim = embedding_dim
        loss, predictions, target_y = train_model(False,False)
        if loss < min_loss:
            min_loss = loss
            best_embedding_dim = embedding_dim
    FLAGS.embedding_dim = tmp


    tmp = FLAGS.num_filters
    best_num_filters = FLAGS.num_filters
    min_loss = 100000000
    list_num_filters = {128, 192, 256, 320}
    for num_filters in list_num_filters:
        FLAGS.num_filters = num_filters
        loss, predictions, target_y = train_model(False,False)
        if loss < min_loss:
            min_loss = loss
            best_num_filters = num_filters
    FLAGS.num_filters = tmp


    FLAGS.embedding_dim = best_embedding_dim
    FLAGS.num_filters = best_num_filters

    list_predictions = list()
    list_filter_sizes = {'1,2,3','2,3,4','3,4,5','4,5,6','2,3,4,5','1,2,3,4','3,4,5,6','1,2,3,4,5','2,3,4,5,6','1,2,3,4,5,6'}
    for filter_sizes in list_filter_sizes:
        FLAGS.filter_sizes = filter_sizes
        loss, predictions, target_y = train_model(False,True)
        list_predictions.append(predictions)

    ensemble_prediction(list_predictions,target_y)


train_model(False,True)

# #auto_param_tuning()
# list_predictions = list()
# list_filter_sizes = {'1,2,3','2,3,4','3,4,5','4,5,6','2,3,4,5','1,2,3,4','3,4,5,6','1,2,3,4,5','2,3,4,5,6','1,2,3,4,5,6'}
# for filter_sizes in list_filter_sizes:
#     FLAGS.filter_sizes = filter_sizes
#     loss, predictions, target_y = train_model(False,True)
#     list_predictions.append(predictions)
#
# ensemble_prediction(list_predictions,target_y)

# list_l2_reg_lambda = {0, 0.2, 0.4, 0.6}
# for l2_reg_lambda in list_l2_reg_lambda:
#     FLAGS.l2_reg_lambda = l2_reg_lambda
#     train_model(False)

# list_embedding_dim = {128,192,256}
# list_num_filters = {128,192,256}
# list_filter_sizes = {'2,3,4', '3,4,5', '2,3,4,5'}
# list_l2_reg_lambda = {0.2}
#
# for embedding_dim in list_embedding_dim:
#     for num_filters in list_num_filters:
#         for filter_sizes in list_filter_sizes:
#             for l2_reg_lambda in list_l2_reg_lambda:
#
#                 FLAGS.embedding_dim = embedding_dim
#                 FLAGS.num_filters = num_filters
#                 FLAGS.filter_sizes = filter_sizes
#                 FLAGS.l2_reg_lambda = l2_reg_lambda
#
#                 train_model(True)





