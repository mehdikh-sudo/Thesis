import h5py
import numpy as np
import torch  # ✅ NEW
from scipy.stats import kendalltau, spearmanr, rankdata  # ✅ NEW
from sklearn.metrics.pairwise import cosine_similarity

from STVT.knapsack import knapsack


def eval_metrics(y_pred, y_true):

    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    # TP = np.sum((y_pred == 1) & (y_true == 1))
    # TN = np.sum((y_pred == 0) & (y_true == 0))
    # FP = np.sum((y_pred == 1) & (y_true == 0))
    # FN = np.sum((y_pred == 0) & (y_true == 1))
    # accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    # accuracy = (TP + TN) / (y_true.shape[0] + 1e-8)

    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return [precision, recall, fscore]


# def select_keyshots(predicted_list, video_number_list,image_name_list,target_list,args):
#     data_path = '/Users/mehdikhosravi/Master/Thesis/STVT-main/STVT/datasets/datasets/'+str(args.dataset).lower()+".h5"
#     data_file = h5py.File(data_path)

#     predicted_single_video = []
#     predicted_single_video_list = []
#     target_single_video = []
#     target_single_video_list = []
#     video_single_list = list(set(video_number_list))
#     eval_arr = []

#     for i in range(len(image_name_list)):
#         if image_name_list[i] == 1 and i!=0:
#             predicted_single_video_list.append(predicted_single_video)
#             target_single_video_list.append(target_single_video)
#             predicted_single_video = []
#             target_single_video = []

#         predictedL = [predicted_list[i]]
#         predicted_single_video += predictedL
#         targetL = list(map(int, str(target_list[i])))
#         target_single_video += targetL

#         if i == len(image_name_list)-1:
#             predicted_single_video_list.append(predicted_single_video)
#             target_single_video_list.append(target_single_video)
#     video_single_list_sort = sorted(video_single_list)
#     True_all_video_len = 0
#     for i in range(len(video_single_list_sort)):
#         index = str(video_single_list_sort[i])
#         video = data_file['video_' + index]
#         fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
#         True_all_video_len += fea_sequencelen

#     for i in range(len(video_single_list_sort)):
#         index = str(video_single_list_sort[i])
#         video = data_file['video_' + index]
#         cps = video['change_points'][:]
#         vidlen = int(cps[-1][1]) + 1
#         weight = video['n_frame_per_seg'][:]
#         fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
#         for ckeck_n in range(len(video_single_list_sort)):
#             dif = True_all_video_len-len(predicted_list)
#             if len(predicted_single_video_list[ckeck_n]) == fea_sequencelen or len(predicted_single_video_list[ckeck_n]) == fea_sequencelen-dif:
#                 pred_score = np.array(predicted_single_video_list[ckeck_n])
#                 up_rate = vidlen//len(pred_score)
#                 # print(up_rate)
#                 break
#         #pred
#         pred_score = upsample(pred_score, up_rate, vidlen)
#         pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
#         _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
#         selected = selected[::-1]
#         key_labels = np.zeros((vidlen,))
#         for i in selected:
#             key_labels[cps[i][0]:cps[i][1]] = 1
#         pred_summary = key_labels.tolist()
#         true_summary_arr_20 = video['user_summary'][:]
#         eval_res = [eval_metrics(pred_summary, true_summary_1) for true_summary_1 in true_summary_arr_20]
#         eval_res = np.mean(eval_res, axis=0).tolist() if args.dataset == "TVSum" else np.max(eval_res, axis=0).tolist()
#         eval_arr.append(eval_res)

#     return eval_arr


def select_keyshots(
    predicted_list, video_number_list, image_name_list, target_list, args
):
    data_path = (
        "/kaggle/input/datasets/mehdikhosravi76/"+str(args.dataset)+"/"
        + str(args.dataset)
        + ".h5"
    )
    data_file = h5py.File(data_path)

    predicted_single_video = []
    predicted_single_video_list = []
    target_single_video = []
    target_single_video_list = []
    video_single_list = list(set(video_number_list))
    eval_arr = []

    # Group frame-level predictions & targets by video
    for i in range(len(image_name_list)):
        if image_name_list[i] == 1 and i != 0:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
            predicted_single_video = []
            target_single_video = []

        predictedL = [predicted_list[i]]
        predicted_single_video += predictedL
        targetL = list(map(int, str(target_list[i])))
        target_single_video += targetL

        if i == len(image_name_list) - 1:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)

    video_single_list_sort = sorted(video_single_list)
    True_all_video_len = 0
    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file["video_" + index]
        fea_sequencelen = (len(video["feature"][:]) // args.sequence) * args.sequence
        True_all_video_len += fea_sequencelen

    # For each video in test set
    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file["video_" + index]
        cps = video["change_points"][:]
        vidlen = int(cps[-1][1]) + 1
        weight = video["n_frame_per_seg"][:]
        fea_sequencelen = (len(video["feature"][:]) // args.sequence) * args.sequence

        # Find the right prediction list matching this video
        for check_n in range(len(video_single_list_sort)):
            dif = True_all_video_len - len(predicted_list)
            if (
                len(predicted_single_video_list[check_n]) == fea_sequencelen
                or len(predicted_single_video_list[check_n]) == fea_sequencelen - dif
            ):
                pred_score = np.array(predicted_single_video_list[check_n])
                up_rate = vidlen // len(pred_score)
                break

        # Upsample to match original frame length
        pred_score = upsample(pred_score, up_rate, vidlen)

        # Segment-level scores for knapsack
        pred_value = np.array([pred_score[cp[0] : cp[1]].mean() for cp in cps])
        _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
        selected = selected[::-1]

        key_labels = np.zeros((vidlen,))
        for j in selected:
            key_labels[cps[j][0] : cps[j][1]] = 1

        pred_summary = key_labels.tolist()
        true_summary_arr_20 = video["user_summary"][:]

        # ---- 🟡 ORIGINAL EVALUATION ----
        eval_res_list = [
            eval_metrics(pred_summary, true_summary_1)
            for true_summary_1 in true_summary_arr_20
        ]
        
        eval_res = (
            np.mean(eval_res_list, axis=0).tolist()
            if args.dataset == "TVSum"
            or args.dataset == "TvSum_Rgb_Flow_Resnet"
            or args.dataset == "TvSum_Rgb_Flow"
            or args.dataset == "TVSum_RFR_matched_10class"
            else np.max(eval_res_list, axis=0).tolist()
        )



        # ---- 🟢 Spearman's ρ & Kendall's τ-b (all datasets) ----
        # Average all annotators' binary selections into one continuous score per frame.
        # e.g. a frame chosen by 12 of 15 annotators gets score 0.8 vs 0.13 for 2/15.
        # This gives frames a proper ranking rather than just two rank levels (0/1).

        rho_list = []
        tau_list = []



        # TVSum: use gtscore (averaged raw 1-5 annotations, continuous, richer ranking).
        # SumMe: no 1-5 scores available; average the 15 binary user_summary annotations
        #        to get a consensus rate per frame (0.0-1.0 in steps of 1/15).
        # Normalisation is NOT needed: Spearman/Kendall are rank-invariant.

        if (
            args.dataset == "TVSum"
            or args.dataset == "TvSum_Rgb_Flow_Resnet"
            or args.dataset == "TvSum_Rgb_Flow"
            or args.dataset == "TVSum_RFR_Normalized"
        ):
            avg_gt_score = video["gtscore"][:]  # shape: (num_frames,), range 1-5
        else:

            avg_gt_score = np.mean(true_summary_arr_20,axis=0)  # shape: (num_frames,) range 0-1
        
        rho_list.append(spearmanr(pred_score, avg_gt_score)[0])
        tau_list.append(kendalltau(rankdata(pred_score), rankdata(avg_gt_score))[0])

        # if np.isnan(rho_final):
        #     rho_final = 0.0
        # if np.isnan(tau_final):
        #     tau_final = 0.0


        # for true_summary_1 in true_summary_arr_20:
        #     rho, _ = spearmanr(pred_score, true_summary_1)
        #     tau, _ = kendalltau(pred_score, true_summary_1)
        #     if np.isnan(rho):
        #         rho = 0
        #     if np.isnan(tau):
        #         tau = 0
        #     rho_list.append(rho)
        #     tau_list.append(tau)

        rho_final = np.mean(rho_list)
        tau_final = np.mean(tau_list)



        # ---- Feature-based BERTScore ----
        def feature_bertscore(candidate_feats, reference_feats):
            """
            Compute a BERTScore-like precision, recall, and F1 between candidate and reference frame features.
            Args:
                candidate_feats: (N_cand, D) torch.Tensor or np.array
                reference_feats: (N_ref, D) torch.Tensor or np.array
            """
            if isinstance(candidate_feats, np.ndarray):
                candidate_feats = torch.from_numpy(candidate_feats).float()
            if isinstance(reference_feats, np.ndarray):
                reference_feats = torch.from_numpy(reference_feats).float()

            # L2 normalize (like BERTScore does on embeddings)
            cand_norm = torch.nn.functional.normalize(candidate_feats, p=2, dim=1)
            ref_norm = torch.nn.functional.normalize(reference_feats, p=2, dim=1)

            # Cosine similarity matrix (N_ref x N_cand)
            sim = torch.matmul(ref_norm, cand_norm.T)  # (N_ref, N_cand)

            # Precision: for each candidate, find the best matching reference frame
            precision = sim.max(dim=0)[
                0
            ].mean()  # max over rows (ref), average over candidates

            # Recall: for each reference, find best matching candidate frame
            recall = sim.max(dim=1)[
                0
            ].mean()  # max over columns (cand), average over refs

            # F1
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            # return precision.item(), recall.item(), f1.item()
            return f1.item()

        # Example usage:

        bertscore_list = []
        for true_summary_1 in true_summary_arr_20:
            pred_idx = np.where(np.array(pred_summary) == 1)[0]
            true_idx = np.where(np.array(true_summary_1) == 1)[0]
            if len(pred_idx) == 0 or len(true_idx) == 0:
                bertscore_list.append(0.0)
            else:
                max_idx = len(video["feature"]) - 1
                pred_idx = np.clip(pred_idx, 0, max_idx)
                true_idx = np.clip(true_idx, 0, max_idx)
                cand_features = video["feature"][pred_idx]
                ref_features = video["feature"][true_idx]
                bertscore_list.append(feature_bertscore(cand_features, ref_features))
        if (
            args.dataset == "TVSum"
            or args.dataset == "TvSum_Rgb_Flow_Resnet"
            or args.dataset == "TvSum_Rgb_Flow_Resnet"
            or args.dataset == "TVSum_RFR_Normalized"
        ):
            bertscore_final = np.mean(bertscore_list)
        else:
            bertscore_final = np.max(bertscore_list)

        # print(f"Feature BERTScore - P: {P:.4f}, R: {R:.4f}, F1: {F1:.4f}")

        ##the first bert score feature:
        # bertscore_list = []
        # for true_summary_1 in true_summary_arr_20:
        #     pred_idx = np.where(np.array(pred_summary) == 1)[0]
        #     true_idx = np.where(np.array(true_summary_1) == 1)[0]
        #     if len(pred_idx) == 0 or len(true_idx) == 0:
        #         bertscore = 0
        #     else:
        #         max_idx = len(video['feature']) - 1
        #         pred_idx = np.clip(pred_idx, 0, max_idx)
        #         true_idx = np.clip(true_idx, 0, max_idx)

        #         pred_feat = video['feature'][pred_idx]
        #         true_feat = video['feature'][true_idx]

        #         sim = cosine_similarity(pred_feat, true_feat)
        #         bertscore = np.mean(sim)
        #     bertscore_list.append(bertscore)
        # if args.dataset == "TVSum" or args.dataset=='TvSum_Rgb_Flow_Resnet'  or args.dataset=='TvSum_Rgb_Flow_Resnet':
        #     bertscore_final = np.mean(bertscore_list)
        # else:
        #      bertscore_final = np.max(bertscore_list)

        # Append metrics — unified for all datasets
        # eval_res indices: [0]=precision [1]=recall [2]=fscore [3]=bert [4]=rho [5]=tau
        
        eval_res += [bertscore_final, rho_final, tau_final]

        eval_arr.append(eval_res)
    # print("pred range:", pred_score.min(), pred_score.max())
    # print("gt range:", avg_gt_score.min(), avg_gt_score.max())

    # print("pred first 20:", np.round(pred_score[:20],3))
    # print("gt   first 20:", np.round(avg_gt_score[:20],3))

    # print("rho =", spearmanr(pred_score, avg_gt_score)[0])
    # print("tau =", kendalltau(pred_score, avg_gt_score)[0])

    # print(pred_score.shape)
    # print(avg_gt_score.shape)


    # idx = np.argsort(avg_gt_score)

    # print("GT sorted:")
    # print(avg_gt_score[idx][:30])

    # print("Pred sorted by GT:")
    # print(pred_score[idx][:30])

    # print("Highest GT:")
    # print(avg_gt_score[idx][-30:])

    # print("Pred at highest GT:")
    # print(pred_score[idx][-30:])

    return eval_arr


def upsample(down_arr, up_rate, vidlen):
    up_arr = np.zeros(vidlen)
    for i in range(len(down_arr)):
        for j in range(up_rate):
            up_arr[i * up_rate + j] = down_arr[i]

    return up_arr
