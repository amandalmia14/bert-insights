import json
import os.path
import sys
import time
import warnings

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import matthews_corrcoef
from tqdm import trange
from transformers import get_linear_schedule_with_warmup, AdamW

sys.path.append(os.getcwd())

from config import get_config_parser
from src.model.BERT import BERT
from src.reproducebase.glue_data import dataloader
from src.utils.misc import seed_experiment, compute_accuracy

pth_sep = os.path.sep


def train(model, iterator, optimizer, scheduler, criterion, device, model_config):
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    start_time = time.time()
    for i, batch in enumerate(iterator):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        input_ids, input_mask, labels = batch

        outputs, _ = model(input_ids, input_mask)

        loss = criterion(outputs, labels)
        acc = compute_accuracy(outputs, labels)

        # delete used variables to free GPU memory
        del batch, input_ids, input_mask, labels
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       model_config["max_grad_norm"])  # Gradient clipping is not in AdamW anymore
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.cpu().item()
        optimizer.zero_grad()
        epoch_accuracy += acc.item() / len(iterator)

    # free GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()

    return epoch_loss / len(iterator), epoch_accuracy, time.time() - start_time


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0
    all_pred = []
    all_label = []

    start_time = time.time()

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            input_ids, input_mask, labels = batch
            # print("input_ids", input_ids)
            # print("input_mask", input_mask)
            # print("labels", labels)
            outputs, _ = model(input_ids, input_mask)

            loss = criterion(outputs, labels)

            # delete used variables to free GPU memory
            del batch, input_ids, input_mask
            epoch_loss += loss.cpu().item()

            # identify the predicted class for each example in the batch
            probabilities, predicted = torch.max(outputs.cpu().data, 1)
            # put all the true labels and predictions to two lists
            all_pred.extend(predicted)
            all_label.extend(labels.cpu())

    accuracy = accuracy_score(all_label, all_pred)
    f1score = f1_score(all_label, all_pred, average='macro')

    predictions = [pred.numpy().item() for pred in all_pred]
    true_labels = [tr.numpy().item() for tr in all_label]
    print(true_labels)
    print(predictions)
    # print("matthews_corrcoef(true_labels, predictions)", matthews_corrcoef(true_labels, predictions))
    math_coef = matthews_corrcoef(true_labels, predictions)
    return epoch_loss / len(iterator), accuracy, math_coef, f1score, time.time() - start_time


if __name__ == '__main__':
    parser = get_config_parser()
    args = parser.parse_args()

    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make that your environment is "
            "running on GPU (e.g. in the Notebook Settings in Google Colab). "
            'Forcing device="cpu".'
        )
        args.device = "cpu"

    if args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and might run out of memory "
            "shortly. You can try setting batch_size=1 to reduce memory usage."
        )

    # Seed the experiment, for repeatability
    device = torch.device(args.device)
    seed_experiment(args.seed)

    train_dataloader, validation_dataloader, tokenizer = dataloader(batch_size=args.batch_size,
                                                                    dataset_name=args.dataset, model=args.model,
                                                                    max_len=args.max_len)
    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')

    model = BERT(model_path=args.model, hidden_size=model_config["hidden_size"]).to(device=device)

    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = num_training_steps * model_config["warmup_proportion"]

    # create the optimizer
    optimizer = AdamW(model.parameters(), lr=model_config["learning_rate"], correct_bias=False)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"])

    # set the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)  # PyTorch

    task_specific_criterion = nn.CrossEntropyLoss()

    bert_checkpoint_dir = os.getcwd() + pth_sep + "src" + pth_sep + "data" + pth_sep + "reproducedmodel" + pth_sep + \
                          args.dataset + pth_sep

    # os.makedirs(bert_checkpoint_dir, exist_ok=True)

    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    train_times, valid_times = [], []

    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss, acc, wall_time = train(model, train_dataloader, optimizer, scheduler, task_specific_criterion,
                                           device, model_config)
        train_losses.append(train_loss)
        train_accs.append(acc)
        train_times.append(wall_time)

        val_loss, val_acc, math_coef, val_f1, wall_time = evaluate(model, validation_dataloader,
                                                                   task_specific_criterion,
                                                                   device)
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        valid_times.append(wall_time)

        model_save_path = bert_checkpoint_dir + pth_sep + str(epoch)
        os.makedirs(bert_checkpoint_dir, exist_ok=True)
        model.bert_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(
            '\n Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation F1: {:.4f}'.format(
                epoch + 1, args.epochs, train_loss, val_loss, val_acc, val_f1))

    print("Fine-tuning completed.")

    # Put model in evaluation mode
    model.eval()

    # Predict
    test_loss, test_acc, test_math_coef, test_f1_score, wall_time = evaluate(model, validation_dataloader,
                                                                             task_specific_criterion,
                                                                             device)
    # Save log if logdir provided
    if args.logdir is not None:
        print(f'Writing training logs to {args.logdir}...')
        # log_dir = args.logdir + pth_sep + args.model + pth_sep + args.dataset + pth_sep
        comfig_name = args.dataset + "_bs_" + str(args.batch_size) + "_ml_" + str(args.max_len)
        os.makedirs(args.logdir, exist_ok=True)
        time_now = time.strftime("%d%m%y_%H:%M")
        with open(os.path.join(args.logdir, time_now + "_" + comfig_name + '_config.json'), 'w') as conf:
            conf.write(str(args))

        with open(os.path.join(args.logdir, time_now + "_" + comfig_name + '_results.json'), 'w') as f:
            f.write(json.dumps(
                {
                    "train_losses": train_losses,
                    "valid_losses": valid_losses,
                    "train_accs": train_accs,
                    "valid_accs": valid_accs,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_math_coef": test_math_coef,
                    "test_f1_score": test_f1_score
                },
                indent=4,
            ))

    print("test_loss", test_loss)
    print("test_acc", test_acc)
    print("test_math_coef", test_math_coef)
    print("f1_score", test_f1_score)
