from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from src.python.modules.bioasqbiobert.run_factoid import *


class QAModel():

    def __init__(self, bert_directory_path, output_dir="data/bioasqbiobert/outputs"):

        self.bert_config_file = os.path.join(bert_directory_path, "bert_config.json")
        self.output_dir = output_dir
        self.vocab_file = os.path.join(bert_directory_path, "vocab.txt")
        self.do_lower_case = True
        self.use_tpu = False
        self.tpu_name = None
        self.tpu_zone = None
        self.gcp_project = None
        self.tpu_cluster_resolver = None
        self.master = None
        self.save_checkpoints_steps = 1000
        self.iterations_per_loop = 1000
        self.num_tpu_cores = 8
        self.num_train_steps = None
        self.num_warmup_steps = None
        self.init_checkpoint = None
        self.learning_rate = 5e-5
        self.train_batch_size = 32
        self.predict_batch_size = 8
        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30

        # tf.logging.set_verbosity(tf.logging.INFO)
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        tf.gfile.MakeDirs(self.output_dir)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        if self.use_tpu and self.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(self.tpu_name, zone=self.tpu_zone,
                                                                                  project=self.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

        run_config = tf.contrib.tpu.RunConfig(
            cluster=self.tpu_cluster_resolver,
            master=self.master,
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,
                num_shards=self.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_tpu)

        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.train_batch_size,
            predict_batch_size=self.predict_batch_size)

    def create_input(self, inp, from_dict=True):
        if from_dict:
            eval_examples = [read_squad_example_from_dict(x) for x in inp]
        else:
            eval_examples = read_squad_examples(input_file=inp, is_training=False)

        eval_writer = FeatureWriter(filename=os.path.join(self.output_dir, "eval.tf_record"), is_training=False)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)
            eval_writer.process_feature(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            output_fn=append_feature)
        eval_writer.close()
        return eval_examples, eval_writer, eval_features

    def predict(self, eval_examples, eval_features, eval_writer, to_dump_to_file=False):
        tf.logging.info("***** Running predictions *****")
        tf.logging.info("  Num orig examples = %d", len(eval_examples))
        tf.logging.info("  Num split examples = %d", len(eval_features))
        tf.logging.info("  Batch size = %d", self.predict_batch_size)

        predict_input_fn = input_fn_builder(
            input_file=eval_writer.filename,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False)

        all_results = []
        for result in self.estimator.predict(predict_input_fn, yield_single_examples=True):
            if len(all_results) % 1000 == 0:
                tf.logging.info("Processing example: %d" % (len(all_results)))
            unique_id = int(result["unique_ids"])
            start_logits = [float(x) for x in result["start_logits"].flat]
            end_logits = [float(x) for x in result["end_logits"].flat]
            all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))

        output_prediction_file = os.path.join(self.output_dir, "predictions.json")
        output_nbest_file = os.path.join(self.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(self.output_dir, "null_odds.json")

        return write_predictions(eval_examples, eval_features, all_results, self.n_best_size, self.max_answer_length,
                                 self.do_lower_case, output_prediction_file, output_nbest_file,
                                 output_null_log_odds_file, to_dump_to_file=to_dump_to_file)

    def predict_from_json(self, input_file, to_dump_to_file=False):
        eval_examples, eval_writer, eval_features = self.create_input(input_file, from_dict=False)
        return self.predict(eval_examples, eval_features, eval_writer, to_dump_to_file=to_dump_to_file)

    def predict_from_dict(self, inp, to_dump_to_file=False):
        if isinstance(inp, dict):  ## if dictionary type
            inp = [inp]
        eval_examples, eval_writer, eval_features = self.create_input(inp)
        return self.predict(eval_examples, eval_features, eval_writer, to_dump_to_file=to_dump_to_file)
