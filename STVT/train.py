import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from STVT.build_dataloader import build_dataloader
from STVT.build_model import build_model
from STVT.build_optimizer import build_optimizer
from STVT.eval import select_keyshots
from STVT.metrics import Metric
from STVT.utils import (
    adjust_learning_rate,
    load_model,
    resume_model,
    save_model,
)
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS (Apple Silicon) > CPU.
# This is computed ONCE, here, and used everywhere in the file (train(), val(),
# train_net()). Do NOT re-assign `device` anywhere else in this file - earlier
# versions of this script re-checked/overwrote `device` inside train_net(),
# which silently forced CPU on any machine without an Apple GPU (e.g. Kaggle).
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

pd_epoch = []
pd_batch_size = []
pd_lr = []
pd_runtime = []
pd_loss = []
pd_val_loss = []  # Add validation loss tracking
pd_F_measure_k = []
# ... add these ...
pd_Spearman_rho_k = []
pd_Best_Spearman_rho_k = []

pd_Kendall_tau_k = []
pd_Best_Kendall_tau_k = []
pd_BERTScore_k = []
pd_Best_F_measure = []


def parse_args():
    parser = argparse.ArgumentParser(description="Image classification")
    parser.add_argument("--roundtimes", type=str, default=1, help="Roundtimes.")
    parser.add_argument("--dataset", default="TVSum", help="Dataset names.")
    parser.add_argument(
        "--test_dataset",
        type=str,
        # default="1,2,11,16,18,20,31,32,35,46",
        # default case is random
        help="The number of test video in the dataset.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        # default=10,
        default=2,
        help="The number of classes in the dataset.",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=16,
        help="The number of sequence.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=40,
        # default=24,
        help="input batch size for training",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=40,
        # default=24,
        help='--test_dataset "5,9,12,23,24" batch size for val',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        # default=130,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--test_epochs",
        type=int,
        default=1,
        help="number of internal epochs to test",
    )
    parser.add_argument("--optim", default="sgd", help="Model names.")

    # parser.add_argument('--optim', default='adamw', help='Optimizer (legacy field).')
    # parser.add_argument('--optimizer', default='adamw', help='Optimizer to use.')

    parser.add_argument("--lr", type=float, default=0.03, help="learning rate")

    # parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')  # Too high, caused overfitting
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument(
        "--warmup_epochs",
        type=float,
        # default=15,
        default=10,
        help="number of warmup epochs",
    )
    # grad_clip #new

    parser.add_argument(
        "--grad_clip",
        type=float,  # default=1.0,
        default=0,
        help="gradient clipping max norm",
    )

    # early stopping # new
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=100,
        help="early stopping patience (epochs)",
    )

    # label smoothing arg # new
    parser.add_argument(
        "--label_smoothing",
        type=float,  # default=0.1,
        default=0,
        help="label smoothing epsilon",
    )

    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight_decay", type=float, default=0.00008, help="weight decay"
    )
    # parser.add_argument(
    #     '--weight_decay', type=float, default=1e-4, help='weight decay'
    # )
    # parser.add_argument(
    #     '--weight_decay', type=float, default=5e-4, help='weight decay'
    # )

    # parser.add_argument('--adam_beta1', type=float, default=0.9, help='Adam beta1')
    # parser.add_argument('--adam_beta2', type=float, default=0.999, help='Adam beta2')
    # parser.add_argument('--eps', type=float, default=1e-8, help='Adam/RMSprop eps')
    # parser.add_argument('--rms_alpha', type=float, default=0.99, help='RMSprop alpha')

    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="To use nesterov or not.",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="how to schedule learning rate",
    )
    parser.add_argument(
        "--resume", action="store_true", default=False, help="Resume training"
    )
    parser.add_argument(
        "--gpu_id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
    )


    parser.add_argument(
    "--data_path",
    type=str,
    default="/kaggle/input/datasets/mehdikhosravi76/summe-rfr-normalized",
    help="Folder containing the dataset's .h5 file",
        )

    # Use parse_known_args() instead of parse_args() so this script does not
    # crash when run inside a Jupyter/Kaggle notebook, which injects its own
    # kernel arguments (e.g. -f /root/.local/.../kernel.json) into sys.argv.
    # A plain script run from the terminal still works exactly the same way.
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unrecognized args (expected in notebooks):", unknown)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    return args


def val(model, val_loader, epoch, args, criterion):

    model.eval()
    if epoch == -1:
        epoch = args.epochs - 1

    global pd_F_measure_k
    global pd_val_loss

    # Track validation loss using Metric class (same as training)
    val_loss_metric = Metric("val_loss")

    with tqdm(total=len(val_loader), desc="Validate Epoch #{}".format(epoch + 1)) as t:
        with torch.no_grad():
            predicted_multi_list = []
            target_multi_list = []
            video_number_list = []
            image_number_list = []
            for data, target, video_number, image_number in val_loader:
                predicted_list = []
                target_list = []
                data = data.to(device)
                output = model(data)
                multi_target = target.permute(1, 0)
                video_number = video_number
                image_number = image_number

                multi_output = output

                # Compute validation loss for this batch (same way as training)
                batch_loss = 0
                for sequence in range(args.sequence):
                    target_seq = multi_target[sequence].to(device)
                    output_seq = multi_output[sequence]
                    ##old:
                    # loss = criterion(output_seq, target_seq)
                    # val_loss_metric.update(loss)  # Update metric for each sequence

                    ##changed
                    loss = criterion(output_seq, target_seq)
                    batch_loss += loss
                    ##

                    predicted_ver2 = []
                    sigmoid = nn.Sigmoid()
                    outputs_sigmoid = sigmoid(output_seq)
                    for s in outputs_sigmoid:
                        predicted_ver2.append(float(s[1]))
                    predicted_list.append(predicted_ver2)
                    target_list.append(target_seq.tolist())

                ##new added (to fix vall loss):
                batch_loss /= args.sequence
                val_loss_metric.update(batch_loss)
                ##

                t.update(1)
                predicted_list = torch.Tensor(predicted_list).permute(1, 0)
                predicted_list = torch.Tensor(predicted_list).reshape(
                    args.val_batch_size * args.sequence
                )
                target_list = torch.Tensor(target_list).permute(1, 0)
                target_list = torch.Tensor(target_list).reshape(
                    args.val_batch_size * args.sequence
                )
                video_number = video_number.reshape(args.val_batch_size * args.sequence)
                image_number = image_number.reshape(args.val_batch_size * args.sequence)
                predicted_multi_list += predicted_list.tolist()
                target_multi_list += target_list.tolist()
                video_number_list += video_number.tolist()
                image_number_list += image_number.tolist()

            predicted_multi_list = [float(i) for i in predicted_multi_list]
            target_multi_list = [int(i) for i in target_multi_list]

            # Get average validation loss (computed same way as training)
            avg_val_loss = val_loss_metric.avg.item()

            eval_res = select_keyshots(
                predicted_multi_list,
                video_number_list,
                image_number_list,
                target_multi_list,
                args,
            )
            # best model by f score:

            fscore_k = 0
            rho_k = 0
            tau_k = 0
            bert_k = 0

            num_videos = len(list(args.test_dataset.split(",")))

            for i in eval_res:
                fscore_k += i[2]  # F-score
                bert_k += i[3]  # BERTScore
                rho_k += i[4]  # Spearman ρ
                tau_k += i[5]  # Kendall τ-b

            fscore_k /= num_videos
            bert_k /= num_videos
            rho_k /= num_videos
            tau_k /= num_videos

            pd_F_measure_k.append(fscore_k)
            pd_val_loss.append(avg_val_loss)
            pd_BERTScore_k.append(bert_k)
            pd_Spearman_rho_k.append(rho_k)
            pd_Kendall_tau_k.append(tau_k)

    save_model(model, args, fscore_k, epoch)

    print("test video number:")
    print(args.test_dataset)
    print("F_measure_k:", fscore_k)
    print("val_loss:", avg_val_loss)
    print("Spearman_rho_k:", rho_k)
    print("Kendall_tau_k:", tau_k)
    print(f"BERTScore_feature_k: {bert_k:.4f}")


def train(model, train_loader, optimizer, criterion, epoch, args):

    global pd_lr
    global pd_loss
    train_loss = Metric("train_loss")
    model.train()
    N = len(train_loader)
    start_time = time.time()

    for batch_idx, (data, target, video_number, image_number) in enumerate(
        train_loader
    ):
        lr_cur = adjust_learning_rate(
            args, optimizer, epoch, batch_idx, N, type=args.lr_scheduler
        )

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        multi_target = target.permute(1, 0)
        multi_output = output
        multi_loss = 0
        for sequence in range(args.sequence):
            target = multi_target[sequence].to(device)
            output = multi_output[sequence]

            loss = criterion(output, target)
            multi_loss += loss

        multi_loss /= args.sequence  # average over N=16 frames (numerically stable)

        multi_loss.backward()

        train_loss.update(multi_loss)

        # Gradient clipping to stabilize training (commented out for now)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        # ----------------------------------------

        if (batch_idx + 1) % 100 == 0:
            # memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            used_time = time.time() - start_time
            eta = used_time / (batch_idx + 1) * (N - batch_idx)
            eta = str(datetime.timedelta(seconds=int(eta)))
            training_state = "  ".join(
                [
                    "Epoch: {}",
                    "[{} / {}]",
                    "eta: {}",
                    "lr: {:.9f}",
                    #'max_mem: {:.0f}',
                    "loss: {:.3f}",
                ]
            )
            training_state = training_state.format(
                epoch + 1,
                batch_idx + 1,
                N,
                eta,
                lr_cur,
                train_loss.avg.item(),
            )
            print(training_state)

        if batch_idx == N - 1:
            pd_lr.append(lr_cur)
            pd_loss.append(train_loss.avg.item())


def train_net(args):
    print("dataset:")
    print(args.dataset)
    print("Init...")
    train_loader, val_loader, In_target = build_dataloader(args)
    total_target = len(train_loader) * args.batch_size * args.sequence
    A = total_target / (total_target - In_target)
    B = total_target / In_target
    model = build_model(args)
    print("Parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = build_optimizer(args, model)

    epoch = 0
    if args.resume:
        epoch = resume_model(model, optimizer, args)

    # NOTE: `device` is the module-level global set once at the top of this
    # file (CUDA > MPS > CPU). We just set the args.cuda/args.mps flags here
    # to match it - we do NOT re-assign `device` itself, since train()/val()
    # read that same global and must stay in sync with wherever the model
    # actually lives.
    args.cuda = device.type == "cuda"
    args.mps = device.type == "mps"

    print("Using device:", device)
    if args.cuda:
        cudnn.benchmark = True

    model.to(device)

    # Use label smoothing if specified
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([A, B]).to(device),
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([A, B]).to(device))

    print("Start training...")

    global pd_epoch
    global pd_batch_size
    global pd_lr
    global pd_runtime
    global pd_loss
    global pd_val_loss
    global pd_F_measure_k
    # added...
    global pd_Spearman_rho_k
    global pd_Kendall_tau_k
    global pd_BERTScore_k

    # Early stopping tracking (removed)
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_f1 = 0
    best_spearman = 0
    best_kendal = 0
    patience = args.early_stop_patience

    while epoch < args.epochs:
        pd_epoch.append(epoch)
        pd_batch_size.append(args.batch_size)
        pd_Best_F_measure.append(best_f1)
        pd_Best_Kendall_tau_k.append(best_kendal)
        pd_Best_Spearman_rho_k.append(best_spearman)
        Stime = time.time()
        train(model, train_loader, optimizer, criterion, epoch, args)
        if (epoch + 1) % args.test_epochs == 0:
            val(model, val_loader, epoch, args, criterion)

            # added best fscore model

            # ... train ...
            # ... evaluate ...
            if len(pd_F_measure_k) > 0:
                current_f1 = pd_F_measure_k[-1]
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    # patience_counter = 0
                save_model(model, args, best_f1, epoch)
                # torch.save(model.state_dict(),args.roundtimes+".pth")

            if len(pd_Kendall_tau_k) and len(pd_Spearman_rho_k) > 0:
                current_spearman = pd_Spearman_rho_k[-1]
                current_kendal = pd_Kendall_tau_k[-1]
                if current_spearman > best_spearman and current_kendal > best_kendal:
                    best_kendal = current_kendal
                    best_spearman = current_spearman

                    patience_counter = 0

                    print(f"**New Best spearman and kendal** : {best_spearman:.4f} and {best_kendal:.4f}")
                    torch.save(model.state_dict,args.roundtimes+"Best_SpKen {best_spearman:.4f}.pth")
                    save_model(model, args, best_spearman, epoch)
                else:
                    patience_counter += 1
                    print(f"-patience {patience_counter}")
                    # if patience_counter >= patience:
                    #     print(f"-Early stopping at epoch {epoch}")
                    #     break

            # added best fscore model

            # # Early stopping check (removed)
            # if len(pd_val_loss) > 0:
            #     current_val_loss = pd_val_loss[-1]
            #     if current_val_loss < best_val_loss:
            #         best_val_loss = current_val_loss
            #         patience_counter = 0
            #         # Save best model state
            #         best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            #         print(f"New best validation loss: {best_val_loss:.4f}")
            #     else:
            #         patience_counter += 1
            #         print(f"Validation loss not improving. Patience: {patience_counter}/{args.early_stop_patience}")

            #         if patience_counter >= args.early_stop_patience:
            #             print(f"Early stopping triggered after {epoch + 1} epochs. Restoring best model...")
            #             if best_model_state is not None:
            #                 model.load_state_dict(best_model_state)
            #             break

        Etime = time.time()
        runtime = str(datetime.timedelta(seconds=int(Etime - Stime)))
        pd_runtime.append(runtime)

        # Make sure the output directory exists (Kaggle's /kaggle/working tree
        # won't have this folder structure unless we create it).
        os.makedirs("./STVT/work_dirs/Record/csv/" + args.dataset, exist_ok=True)


        ddict = {
            "test_dataset": args.test_dataset,
            "weight_decay": args.weight_decay,
            "epoch": pd_epoch,
            "Batch_size": pd_batch_size,
            "lr": pd_lr,
            "runtime": pd_runtime,
            "train_loss": pd_loss,
            "val_loss": pd_val_loss,
            "F_measure_k": pd_F_measure_k,
            "Best_F_measure": pd_Best_F_measure,
            "Spearman_rho_k": pd_Spearman_rho_k,
            "Best_Spearman_rho_k": pd_Best_Spearman_rho_k,
            "Kendall_tau_k": pd_Kendall_tau_k,
            "Best_Kendall_tau_k": pd_Best_Kendall_tau_k,
            "BERTScore_k": pd_BERTScore_k,
        }
        dataframe = pd.DataFrame(ddict)
        csv_path = (
            # "./STVT/work_dirs/Record/csv/"
            "/content/drive/MyDrive"
            + args.dataset
            + "/Record_"
            + str(args.roundtimes)
            + ".csv"
        )
        
        dataframe.to_csv(csv_path, index=False, sep=",")

        epoch += 1


if __name__ == "__main__":
    args = parse_args()
    train_net(args)
