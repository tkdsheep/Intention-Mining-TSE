import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sys import stdin
import csv



class_names = ('information giving', 'information seeking', 'feature request', 'solution proposal', 'problem discovery', 'aspect evaluation', 'others')



# Checkpoint Parameters
tf.flags.DEFINE_string("checkpoint_dir", "runs/docker tensorflow bootstrap vscode/checkpoint-256-2,3,4,5-192-0.5-0.2-64-30", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Map data into vocabulary
vocab_path = "runs/docker tensorflow bootstrap vscode/vocab"
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)



print FLAGS.checkpoint_dir
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
print ("current using model: " + FLAGS.checkpoint_dir)


#checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        phase_train = graph.get_operation_by_name("phase_train").outputs[0]
        conv = graph.get_operation_by_name("conv-maxpool-3").outputs


        # Tensors we want to evaluate
        predictions_tensor = graph.get_operation_by_name("output/predictions").outputs[0]

        print "input a sentence and the model will predict a label for you ^__^"
        while True:

            text = [stdin.readline()]
            x_test = np.array(list(vocab_processor.transform(text)))
            predictions = sess.run(predictions_tensor, {input_x: x_test, dropout_keep_prob: 1.0, phase_train: False})
            print predictions, class_names[predictions[0]]
            print ""
            print conv



