""" CHOWDER implementation main training script. """

# Author: François Joubert <fxa.joubert@gmail.com>
# License: MIT

import argparse
import datetime
import logging
from pathlib import Path

import tensorboardX
import torch
from torch.utils.data import DataLoader

from chowder.data import load_labels_as_dict, load_slide_data_as_dict, save_prediction_on_disk
from chowder.dataset import ChowderDataset
from chowder.model import ChowderModel
from chowder.training import TrainingParams, train, evaluate

logger = logging.getLogger(__name__)

# Config

DEFAULT_ROOT_DATA_FOLDER = Path(__file__).parent.parent / 'data'
TRAIN_LABELS_FILENAME = 'train_output.csv'
TEST_LABELS_FILENAME = 'test_output.csv'

TRAIN_INPUT_FEATURES_FOLDER = Path('train_input') / 'resnet_features'
TEST_INPUT_FEATURES_FOLDER = Path('test_input') / 'resnet_features'
EXPERIMENT_FOLDER = Path(__file__).parent / 'experiments'

TEST_OUTPUT_CSV_FILENAME = 'test_output.csv'

NUM_TILES = 1000
NUM_FEATURES = 2048
LR = 1e-3
NUM_EPOCHS = 30
TRAIN_BATCH_SIZE = 10
R = 5


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='''Performs training and evaluation on a given data folder. This folder should follow a certain
         format that the config is complying to.''')
    parser.add_argument('--data_folder', type=str, default=DEFAULT_ROOT_DATA_FOLDER)
    args = parser.parse_args()
    root_data_folder = Path(args.data_folder)

    if not root_data_folder.exists():
        raise SystemExit("The provided data folder does not exist.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Training using {device}")

    # Get model
    model = ChowderModel(NUM_FEATURES, R=R)

    # Create experiment directory
    exp_dir = EXPERIMENT_FOLDER
    exp_dir.mkdir(exist_ok=True)

    # Create the tensorboard folder from date/time to log training/validation plots
    current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")

    training_dir = exp_dir / (current_date + '_' + model.__class__.__qualname__)
    tb_writer = tensorboardX.SummaryWriter(log_dir=training_dir)

    # Create Dataset

    train_labels_dict = load_labels_as_dict(root_data_folder / TRAIN_LABELS_FILENAME)
    train_filename_seq = list((root_data_folder / TRAIN_INPUT_FEATURES_FOLDER).glob('*'))
    train_slide_data_dict = load_slide_data_as_dict(train_filename_seq)

    train_dataset = ChowderDataset(train_labels_dict, train_slide_data_dict, NUM_TILES, NUM_FEATURES)

    test_labels_dict = load_labels_as_dict(root_data_folder / TEST_LABELS_FILENAME)
    test_filename_seq = list((root_data_folder / TEST_INPUT_FEATURES_FOLDER).glob('*'))
    test_slide_data_dict = load_slide_data_as_dict(test_filename_seq)

    test_dataset = ChowderDataset(test_labels_dict, test_slide_data_dict, NUM_TILES, NUM_FEATURES)

    train_data_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Training parameters

    training_params = TrainingParams(
        device=str(device),
        log_interval=5,
        num_epochs=NUM_EPOCHS,
        eval_interval=1,
        learning_rate=LR
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_params.learning_rate)
    loss_fn = torch.nn.NLLLoss()

    best_auc_score = -1

    for epoch in range(0, training_params.num_epochs):
        average_loss = train(training_params, model, train_data_loader, optimizer, loss_fn, epoch)
        tb_writer.add_scalar("%s_loss" % 'train', average_loss, epoch)

        if epoch % training_params.eval_interval == 0:
            average_loss, auc_score, predictions = evaluate(training_params, model, test_data_loader, loss_fn)
            tb_writer.add_scalar("%s_loss" % 'test', average_loss, epoch)
            tb_writer.add_scalar("%s_acc" % 'test', auc_score, epoch)

            # Overwriting the saved output each time in case of early stopping
            if auc_score > best_auc_score:
                logger.info("Best auc_score found, saving checkpoint\n")
                best_auc_score = auc_score
                torch.save(
                    model.state_dict(),
                    Path(training_dir / f"best_auc_score_{model.__class__.__name__}.pt")
                )
                save_prediction_on_disk(training_dir / TEST_OUTPUT_CSV_FILENAME, predictions, test_dataset.id_list)


if __name__ == '__main__':
    main()
