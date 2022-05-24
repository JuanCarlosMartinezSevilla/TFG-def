from config import Config
from data import DataGenerator
from model import build_model
from evaluator import ModelEvaluator
import utils as U
import argparse


def main(args):
    dg = DataGenerator(dataset_list_path=args.train,
                       batch_size=Config.batch_size,
                       num_channels=Config.num_channels,
                       width_reduction=Config.width_reduction)

    model_tr, model_pr = build_model(vocabulary_size=len(dg.w2i))

    X_val, Y_val, _, _ = U.parse_lst(args.validation)
    evaluator_val = ModelEvaluator([X_val, Y_val])

    X_test, Y_test, _, _ = U.parse_lst(args.test)
    evaluator_test = ModelEvaluator([X_test, Y_test])

    
    best_ser_val = 100
    data = []

    for super_epoch in range(Config.epochs):
        print("Epoch {}".format(super_epoch))
        model_tr.fit(dg,
                     steps_per_epoch=len(dg.X)//Config.batch_size,
                     #steps_per_epoch=100,
                     epochs=1,
                     verbose=0)

        print("\tEvaluating...")
        ser_val = evaluator_val.eval(model_pr, dg.i2w)
        ser_test = evaluator_test.eval(model_pr, dg.i2w)
        print("\tEpoch {}\t{:.2f}\t{:.2f}".format(super_epoch, ser_val, ser_test))

        if ser_val < best_ser_val:
            print("\tSER improved from {} to {} --> Saving model.".format(best_ser_val, ser_val))
            best_ser_val = ser_val
            model_pr.save_weights("model_weights.h5")
        data.append([super_epoch, ser_val])
    print(data)
    with open('results.txt', 'w') as f:
        for elem in data:
            f.write(elem + '\n')
    print("Final")

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-tr', '--train', action='store', required=True,
                        help='List of training samples.')
    parser.add_argument('-val', '--validation', action='store', required=True,
                        help='List of validation samples.')
    parser.add_argument('-ts', '--test', action='store', required=True,
                        help='List of test samples.')
    return parser


if __name__ == '__main__':

    parser = build_argument_parser()
    args = parser.parse_args()

    main(args)
