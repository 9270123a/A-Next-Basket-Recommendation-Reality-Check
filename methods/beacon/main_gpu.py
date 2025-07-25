import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import os
import warnings
import os
import utils
import models
import procedure
from utils import *

# 停用 Eager Execution（若你在 TF1 環境，需要此行才可使用舊式 Session）
tf.compat.v1.disable_eager_execution()

#------------------------------------------------------------------------------#
# 只使用 tf.flags (舊版 TensorFlow 1.x 內建的旗標系統)
#------------------------------------------------------------------------------#

tf.flags.DEFINE_string("dataset", 'CRSP', "The input data directory (default: None)")
tf.flags.DEFINE_string("result_dir", 'temp_result', "The input data directory (default: None)")
tf.flags.DEFINE_integer("foldk", 0, "The input data directory (default: None)")
tf.flags.DEFINE_string("device_id", None, "GPU device is to be used in training (default: None)")
tf.flags.DEFINE_integer("seed", 2, "Seed value for reproducibility (default: 89)")

# Model hyper-parameters
tf.flags.DEFINE_string("data_dir", None, "The input data directory (default: None)")
tf.flags.DEFINE_string("output_dir", "results", "The output directory (default: None)")
tf.flags.DEFINE_string("tensorboard_dir", "tf_folder", "The tensorboard directory (default: None)")

tf.flags.DEFINE_integer("emb_dim", 64, "The dimensionality of embedding (default: 2)")
tf.flags.DEFINE_integer("rnn_unit", 64, "The number of hidden units of RNN (default: 4)")
tf.flags.DEFINE_integer("nb_hop", 1, "The number of neighbor hops (default: 1)")
tf.flags.DEFINE_float("alpha", 0.5, "The reguralized hyper-parameter (default: 0.5)")
tf.flags.DEFINE_integer("matrix_type", 1, "The type of adjacency matrix (0=zero,1=real,default:1)")

# Training hyper-parameters
tf.flags.DEFINE_integer("nb_epoch", 15, "Number of epochs (default: 15)")
tf.flags.DEFINE_integer("early_stopping_k", 5, "Early stopping patience (default: 5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate (default: 0.001)")
tf.flags.DEFINE_float("epsilon", 1e-8, "The epsilon threshold in training (default: 1e-8)")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout keep probability for RNN (default: 0.3)")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size (default: 32)")
tf.flags.DEFINE_integer("display_step", 10, "Show loss/acc for every display_step batches (default: 10)")
tf.flags.DEFINE_string("rnn_cell_type", "LSTM", "RNN Cell Type like LSTM, GRU, etc. (default: LSTM)")
tf.flags.DEFINE_integer("top_k", 10, "Top K Accuracy (default: 10)")
tf.flags.DEFINE_boolean("train_mode", False, "Turn on/off the training mode (default: False)")
tf.flags.DEFINE_boolean("tune_mode", False, "Turn on/off the tunning mode (default: False)")
tf.flags.DEFINE_boolean("prediction_mode", False, "Turn on/off the testing mode (default: False)")

# 從 tf.flags 取出參數
config = tf.flags.FLAGS

def main(_):
    """主程式進入點 (搭配 tf.app.run)。"""
    warnings.filterwarnings('ignore')

    print("---------------------------------------------------")
    print("目前 dataset=", config.dataset)
    print("SeedVal =", str(config.seed))
    print("\nParameters:", str(config.__len__()))
    for iterVal in config.__iter__():
        print(" + {}={}".format(iterVal, config.__getattr__(iterVal)))
    print("Tensorflow version:", tf.__version__)
    print("---------------------------------------------------")

    # 指定 GPU 環境變數
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # For reproducibility
    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)
    tf.logging.set_verbosity(tf.logging.ERROR)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.log_device_placement = False

    # ----------------------- MAIN PROGRAM -----------------------
    data_dir = config.data_dir
    temp_dir = config.result_dir
    dataset = config.dataset
    foldk = config.foldk

    data_file = rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\mergeddataset\CRSP_merged.json'
    keyset_path = rf'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check\keyset\CRSP_keyset_0.json'

    output_dir = config.output_dir
    tensorboard_dir = config.tensorboard_dir

    print("***************************************************************************************")
    print("Output Dir:", output_dir)
    print("@Create directories")
    create_folder(f'{output_dir}/models/{dataset}_{foldk}')
    create_folder(output_dir + "/pred")
    if tensorboard_dir is not None:
        create_folder(tensorboard_dir)

    print("@Load train,validate&test data")
    training_instances, train_uids = get_instances(data_file, keyset_path, mode='train')
    nb_train = len(training_instances)
    print(" + Total training sequences:", nb_train)

    validate_instances, val_uids = get_instances(data_file, keyset_path, mode='val')
    nb_validate = len(validate_instances)
    print(" + Total validating sequences:", nb_validate)

    testing_instances, test_uids = get_instances(data_file, keyset_path, mode='test')
    nb_test = len(testing_instances)
    print(" + Total testing sequences:", nb_test)

    print("@Build knowledge")
    MAX_SEQ_LENGTH, item_dict, rev_item_dict, item_probs = build_knowledge(
        training_instances, validate_instances, testing_instances
    )

    print("#Statistic")
    NB_ITEMS = len(item_dict)
    print(" + Maximum sequence length:", MAX_SEQ_LENGTH)
    print(" + Total items:", NB_ITEMS)

    matrix_type = config.matrix_type
    if matrix_type == 0:
        print("@Create an zero adjacency matrix")
        adj_matrix = create_zero_matrix(NB_ITEMS)
    else:
        print("@Load the normalized adjacency matrix")


        matrix_fpath = os.path.join(
            r'C:\Users\user\NBR-Project\A-Next-Basket-Recommendation-Reality-Check',
            'temp_result',
            f'{dataset}_{foldk}_adj_matrix',
            f'r_matrix_{config.nb_hop}w.npz'   # 例：r_matrix_1w.npz
        )
        adj_matrix = sp.load_npz(matrix_fpath)
        print(" + Real adj_matrix has been loaded from", matrix_fpath)

    print("@Compute #batches in train/validation/test")
    total_train_batches = compute_total_batches(nb_train, config.batch_size)
    total_validate_batches = compute_total_batches(nb_validate, config.batch_size)
    total_test_batches = compute_total_batches(nb_test, config.batch_size)
    print(" + #batches in train", total_train_batches)
    print(" + #batches in validate", total_validate_batches)
    print(" + #batches in test", total_test_batches)

    model_dir = f'{output_dir}/models/{dataset}_{foldk}'

    # ----------------------- TRAINING -----------------------
    if config.train_mode:
        with tf.Session(config=gpu_config) as sess:
            print("================== TRAINING ====================")
            # Create data generator
            train_generator = seq_batch_generator(
                training_instances, item_dict, config.batch_size, uids=train_uids
            )
            validate_generator = seq_batch_generator(
                validate_instances, item_dict, config.batch_size, is_train=False, uids=val_uids
            )
            test_generator = seq_batch_generator(
                testing_instances, item_dict, config.batch_size, is_train=False, uids=test_uids
            )

            print(" + Initialize the network")
            net = models.Beacon(
                sess, config.emb_dim, config.rnn_unit, config.alpha, MAX_SEQ_LENGTH,
                item_probs, adj_matrix, config.top_k, config.batch_size,
                config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate
            )

            print(" + Initialize parameters")
            sess.run(tf.global_variables_initializer())

            print("@Start training")
            procedure.train_network(
                sess, net, train_generator, validate_generator, config.nb_epoch,
                total_train_batches, total_validate_batches, config.display_step,
                config.early_stopping_k, config.epsilon, tensorboard_dir, model_dir,
                test_generator, total_test_batches
            )

        tf.reset_default_graph()

    # ----------------------- PREDICTION / TUNING -----------------------
    if config.prediction_mode or config.tune_mode:
        with tf.Session(config=gpu_config) as sess:
            print(" + Initialize the network")
            net = models.Beacon(
                sess, config.emb_dim, config.rnn_unit, config.alpha, MAX_SEQ_LENGTH,
                item_probs, adj_matrix, config.top_k, config.batch_size,
                config.rnn_cell_type, config.dropout_rate, config.seed, config.learning_rate
            )

            print(" + Initialize parameters")
            sess.run(tf.global_variables_initializer())

            print("===============================================\n")
            print("@Restore the model from", model_dir)
            # Reload the best model
            saver = tf.train.Saver()
            recent_dir = recent_model_dir(model_dir)
            saver.restore(sess, model_dir + "/" + recent_dir + "/model.ckpt")
            print("Model restored from file:", recent_dir)

            # Tuning mode
            if config.tune_mode:
                print("@Start tunning")
                validate_generator = seq_batch_generator(
                    validate_instances, item_dict, config.batch_size, False, uids=val_uids
                )
                procedure.tune(
                    net, validate_generator, total_validate_batches,
                    config.display_step, output_dir + "/topN/val_recall.txt"
                )

            # Prediction mode
            if config.prediction_mode:
                print("@Start generating prediction")
                test_generator = seq_batch_generator(
                    testing_instances, item_dict, config.batch_size, False, uids=test_uids
                )
                pred_path = f'{output_dir}/pred/{dataset}_pred{foldk}.json'
                procedure.generate_prediction(
                    net, test_generator, total_test_batches, config.display_step,
                    rev_item_dict, pred_path
                )

        tf.reset_default_graph()

#------------------------------------------------------------------------------#
# tf.app.run() 會先解析 tf.flags，再呼叫上面定義的 main(_)
#------------------------------------------------------------------------------#
if __name__ == '__main__':
    tf.app.run(main)
