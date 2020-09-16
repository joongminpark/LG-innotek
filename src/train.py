import logging
import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm, trange
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from datasets import LGDatasets
from model import LG_model
from utils import (
    rotate_checkpoints,
    print_result,
    ResultWriter,
)


logger = logging.getLogger(__name__)


def train(args, train_dataset, model):
    result_writer = ResultWriter(args.eval_results_dir)

    # How to sampling
    # weighted sampling
    if args.sample_criteria == "abnormal":
        counts = train_dataset.labels
        np_counts = np.array(counts)
        # num of abnormal & normal
        num_abnormal = len(np_counts[np_counts == 1])
        num_normal = len(np_counts) - num_abnormal
        weights = [1.0 / num_abnormal if l == 1 else 1.0 / num_normal for l in counts]
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    # Random sampling
    else:
        sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler,)
    t_total = len(train_dataloader) * args.num_train_epochs


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, t_total)

    criterion = torch.nn.CrossEntropyLoss()

    # Train!
    logger.info("***** Running LG abnormal test *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    tr_loss, logging_loss = 0.0, 0.0

    best_loss = 1e10

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch",)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            (
                img_x,
                img_y,
                img_z,
                label,
            ) = list(map(lambda x: x.to(args.device), batch))
            model.train()
            
            # sentiment input
            logit = model(
                img_x,
                img_y,
                img_z,
            )

            loss = criterion(logit, label)
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

        # 1 epoch -> evaluate with validation data
        results, cm = evaluate(args, args.eval_data_file, model)
        output_eval_file = os.path.join(args.output_dir, "eval_results_pos.txt")
        print_result(output_eval_file, results, cm)
        
        trn_loss = (tr_loss - logging_loss) / len(train_dataloader)
        logger.info("  Now training loss : %s", trn_loss)
        logging_loss = tr_loss

        if best_loss > results["loss"]:
            best_acc = results["accuracy"]
            best_loss = results["loss"]

            output_dir = os.path.join(args.output_dir, "best_model/")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving best model to %s", output_dir)

            result_path = os.path.join(output_dir, "best_results.txt")
            print_result(result_path, results, cm, off_logger=True)

            result_writer.update(args, **results)

        logger.info("  best val loss : %s", best_loss)
        logger.info("  best accuracy : %s", best_acc)
        
        checkpoint_prefix = "checkpoint"
        # Save model checkpoint
        output_dir = os.path.join(
            args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
        )
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        rotate_checkpoints(args, checkpoint_prefix)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)


    return global_step, tr_loss / global_step


def evaluate(args, dataset_path, model, prefix=""):
    eval_dataset = LGDatasets(dataset_path)
    sampler = SequentialSampler(eval_dataset)
    logger.info(len(eval_dataset))
    eval_dataloader = DataLoader(
        eval_dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=4,
    )
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    loss_fct = torch.nn.NLLLoss()

    model.eval()
    total_anormal_preds = []
    total_anormal_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        (
            img_x,
            img_y,
            img_z,
            label,
        ) = list(map(lambda x: x.to(args.device), batch))

        with torch.no_grad():
            anormaly_logit = model(
                img_x,
                img_y,
                img_z,
            )

            anormaly_score = anormaly_logit.log_softmax(dim=-1)
            anormaly_loss = loss_fct(anormaly_score, label)

            # calculate validation loss
            eval_loss += anormaly_loss.mean().item()
        nb_eval_steps += 1

        anormaly_preds = torch.softmax(anormaly_score, dim=-1).detach().argmax(axis=-1)
        anormaly_labels = label.detach().cpu()

        total_anormal_preds.append(anormaly_preds)
        total_anormal_labels.append(anormaly_labels)

    total_anormal_preds = torch.cat(total_anormal_preds).tolist()
    total_anormal_labels = torch.cat(total_anormal_labels).tolist()

    eval_loss = eval_loss / nb_eval_steps
    result = {'loss': eval_loss}

    # normal(0), anormal(1)
    label_list = list(range(2))

    cm = confusion_matrix(total_anormal_labels, total_anormal_preds, labels=label_list)

    # calculate accuracy
    acc = sum([i == j for i, j in zip(total_anormal_labels, total_anormal_preds)]) / len(
        total_anormal_labels
    )
    result["accuracy"] = acc

    return result, cm


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="Batch size for training.",
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Batch size for training.",
    )
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="../exp/results.csv",
        help="Directory for evaluation report result (for experiments)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="set logging level DEBUG",
    )
    parser.add_argument(
        "--sample_criteria",
        type=str,
        default=None,
        help="Criteria of sampling(abnormal)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )


    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    train_dataset = LGDatasets(args.train_data_file)

    model = LG_model()
    model.to(args.device)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(
                    glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True)
                )
            )
            logging.getLogger("evaluation").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model.load_state_dict(torch.load(checkpoint + "/pytorch_model.bin"))
            model.to(args.device)

            result = evaluate(args, args.eval_data_file, model, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result[0].items())

            results.update(result)


if __name__ == "__main__":
    main()
