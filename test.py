# # eval_quantile_sensitivity.py
# """
# 完整评估脚本：
# - 将预测负值裁为 0
# - 计算并输出：RMSE/Brier、Traditional AP (bin=0.1)、Weighted AP（global & per-sample mean）
#   Quantile-threshold AP 敏感性分析（quantile 0.3~0.9 step 0.05）
#   Coverage@top% / HitRate@PAI@top%（top1%,5%,10%）
# - 输出 per-sample mean 与 global aggregated 两种结果
# - 绘图并保存到 ./Result/Eval/
# """
# import os
# import math
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch.utils.data import DataLoader
# from sklearn.metrics import average_precision_score
# from scipy.ndimage import binary_dilation

# from RiskmapDataset import RiskMapTest
# from DiffResCrime import STFusionNet, DiffRes

# # ----- 配置 -----
# CHECKPOINT_PATH = "./Result/Output/NY_gamma_0.1.pth"
# # CHECKPOINT_PATH = "./Result/Output/best_network_with_early_stopping.pth"
# SAVE_DIR = "./Result/Eval/temp"
# os.makedirs(SAVE_DIR, exist_ok=True)

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 16
# N_FEAT = 256
# N_T = 15
# # WS_TEST = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]    # list of guide_w to evaluate
# # WS_TEST = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# WS_TEST = [0.0]
# CLIP_NEGATIVE = True

# # quantile sensitivity range (0.3 -> 0.9 inclusive, step 0.05)
# QUANTILES = np.arange(0.3, 0.91, 0.05)

# # top% list for coverage/hit/PAI
# TOP_PERCENTS = [0.01, 0.05, 0.10]

# # fixed bin threshold for "traditional" AP and HitRate/PAI
# # BIN_THRESH = 0.1
# BIN_THRESH = 0.0

# # coverage curve resolution (for plotting)
# COVERAGE_TOP_PCT_RANGE = np.linspace(0.005, 0.20, 80)  # 0.5% ~ 20%

# # ----------------- 工具函数 -----------------
# def clamp_neg_to_zero(pred):
#     return np.clip(np.array(pred, dtype=float), 0.0, None)

# def rmse_map(pred_map, gt_map):
#     mask = np.isfinite(pred_map) & np.isfinite(gt_map)
#     if not mask.any(): return float('nan')
#     return float(np.sqrt(np.mean((pred_map[mask] - gt_map[mask])**2)))

# def brier_map(pred_map, gt_map):
#     mask = np.isfinite(pred_map) & np.isfinite(gt_map)
#     if not mask.any(): return float('nan')
#     return float(np.mean((pred_map[mask] - gt_map[mask])**2))

# def traditional_ap(pred_map, gt_map, bin_thresh=BIN_THRESH):
#     pred = pred_map.flatten()
#     gt_bin = (gt_map.flatten() > bin_thresh).astype(int)
#     if gt_bin.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(gt_bin, pred))

# def relaxed_ap(pred_map, gt_map, dilation_radius=1, bin_thresh=BIN_THRESH):
#     gt_bin = (gt_map > bin_thresh).astype(np.uint8)
#     structure = np.ones((2*dilation_radius+1, 2*dilation_radius+1), dtype=np.uint8)
#     gt_dilated = binary_dilation(gt_bin, structure=structure).astype(int)
#     if gt_dilated.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(gt_dilated.flatten().astype(int), pred_map.flatten()))

# def weighted_average_precision(pred_map, label_map):
#     pred = pred_map.flatten()
#     rel = label_map.flatten().astype(float)
#     total_rel = rel.sum()
#     if total_rel == 0:
#         return float('nan')
#     idx = np.argsort(pred)[::-1]
#     rel_sorted = rel[idx]
#     cumsum_rel = np.cumsum(rel_sorted)
#     ranks = np.arange(1, len(rel_sorted) + 1)
#     precision_at_r = cumsum_rel / ranks
#     wap = (precision_at_r * rel_sorted).sum() / total_rel
#     return float(wap)

# def coverage_topk(pred_map, label_map, top_percent):
#     pred = pred_map.flatten()
#     rel = label_map.flatten().astype(float)
#     N = len(pred)
#     k = max(1, int(N * top_percent))
#     idx = np.argsort(pred)[::-1][:k]
#     total_rel = rel.sum()
#     if total_rel == 0:
#         return float('nan')
#     return float(rel[idx].sum() / total_rel)

# def hit_rate_and_pai(pred_map, gt_map, top_percent, bin_thresh=BIN_THRESH):
#     pred = pred_map.flatten()
#     gt = gt_map.flatten()
#     gt_bin = (gt > bin_thresh).astype(int)
#     N = len(pred)
#     k = max(1, int(N * top_percent))
#     idx = np.argsort(pred)[::-1][:k]
#     c_total = gt_bin.sum()
#     if c_total == 0:
#         return float('nan'), float('nan')
#     c_pred = gt_bin[idx].sum()
#     hit = float(c_pred) / float(c_total)
#     area_ratio = float(k) / float(N)
#     pai = (float(c_pred) / float(c_total)) / area_ratio if area_ratio > 0 else float('nan')
#     return hit, pai

# # quantile-threshold AP (per-sample)
# def quantile_ap_per_sample(pred_map, gt_map, quantile):
#     """
#     在单个样本级别：以该样本的 gt_map 的 quantile 作阈值，二值化 gt 后计算 AP
#     返回 AP（或 nan）
#     """
#     gt_flat = gt_map.flatten()
#     if np.all(gt_flat == gt_flat[0]):
#         if gt_flat[0] == 0:
#             return float('nan')
#         else:
#             # 全正样本 -> AP 可以计算为 average_precision_score(all 1s, pred)
#             y_true = np.ones_like(gt_flat, dtype=int)
#             return float(average_precision_score(y_true, pred_map.flatten()))
#     threshold = np.quantile(gt_flat, quantile)
#     y_true = (gt_flat >= threshold).astype(int)
#     if y_true.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(y_true, pred_map.flatten()))

# # quantile-threshold AP (global: 所有样本合并后用全局 gt quantile 作阈值)
# def quantile_ap_global(pred_list, gt_list, quantile):
#     preds = np.concatenate([p.flatten() for p in pred_list])
#     gts = np.concatenate([g.flatten() for g in gt_list])
#     mask = np.isfinite(preds) & np.isfinite(gts)
#     preds = preds[mask]; gts = gts[mask]
#     if preds.size == 0:
#         return float('nan')
#     if np.all(gts == gts[0]):
#         if gts[0] == 0:
#             return float('nan')
#         else:
#             return float(average_precision_score(np.ones_like(gts, dtype=int), preds))
#     thresh = np.quantile(gts, quantile)
#     y_true = (gts >= thresh).astype(int)
#     if y_true.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(y_true, preds))


# # ----------------- 主流程 -----------------
# def main():
#     # load model
#     # fusion_net = STFusionNet(in_channels=1, n_feat=N_FEAT, poi_channels=13)
#     fusion_net = STFusionNet(
#     in_channels=1,
#     n_feat=N_FEAT,        # 256
#     poi_channels=13,
#     x_dim=2048            # 1024
#     )
#     diffres_model = DiffRes(backbone=fusion_net, n_T=N_T, device=DEVICE, drop_prob=0.1)

#     if not os.path.exists(CHECKPOINT_PATH):
#         raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
#     diffres_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
#     diffres_model.to(DEVICE)
#     diffres_model.eval()
#     print("Model loaded ->", CHECKPOINT_PATH)
#     print("Device:", DEVICE)

#     # dataloader
#     dataset = RiskMapTest()
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=False)
#     print("Dataset size:", len(dataset), "batches:", len(dataloader))

#     # containers
#     per_sample_metrics = {w: {'rmse': [], 'brier': [], 'trad_ap': [], 'relaxed_ap_r1': [], 'weighted_ap_per_sample': [],
#                              'coverage_1%': [], 'coverage_5%': [], 'coverage_10%': [],
#                              'hit_1%': [], 'pai_1%': [], 'hit_5%': [], 'pai_5%': [], 'hit_10%': [], 'pai_10%': []}
#                           for w in WS_TEST}
#     all_preds = {w: [] for w in WS_TEST}
#     all_gts = {w: [] for w in WS_TEST}

#     # 记录 quantile AP across samples (per-sample mean) for each w and quantile
#     quantile_results_per_sample_mean = {w: {q: [] for q in QUANTILES} for w in WS_TEST}

#     # evaluation loop
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(dataloader):
#             filenames = batch['file_name']
#             labels = batch['riskmap_obs']             # Tensor (B,1,H,W)
#             cond_map = batch['map']
#             cond_satellite = batch['satellite']
#             history = batch['riskmap_his']
#             poi = batch['poi']
 
#             # 合并卫星和地图条件 [B, H, W, 6]
#             cond_combined = torch.cat([cond_satellite, cond_map], dim=-1)

#             labels = labels.to(DEVICE)
#             cond_combined = cond_combined.to(DEVICE)
#             history = history.to(DEVICE)
#             poi = poi.to(DEVICE)

#             B = labels.shape[0]
#             for w in WS_TEST:
#                 try:
#                     x_gen = diffres_model.sample(n_sample=B, size=(1, 16, 16),
#                                         cond=cond_combined, x_hist=history, poi=poi, guide_w=w)
#                 except Exception as e:
#                     print(f"Sampling error in batch {batch_idx} w={w}: {e}")
#                     continue

#                 if isinstance(x_gen, torch.Tensor):
#                     x_gen = x_gen.cpu().numpy()

#                 for i in range(B):
#                     pred = x_gen[i, 0].astype(float)
#                     gt = labels[i, 0].cpu().numpy().astype(float)

#                     if CLIP_NEGATIVE:
#                         pred = clamp_neg_to_zero(pred)

#                     # store for global metrics
#                     all_preds[w].append(pred.copy())
#                     all_gts[w].append(gt.copy())

#                     # per-sample metrics
#                     per_sample_metrics[w]['rmse'].append(rmse_map(pred, gt))
#                     per_sample_metrics[w]['brier'].append(brier_map(pred, gt))
#                     per_sample_metrics[w]['trad_ap'].append(traditional_ap(pred, gt, BIN_THRESH))
#                     per_sample_metrics[w]['relaxed_ap_r1'].append(relaxed_ap(pred, gt, dilation_radius=1, bin_thresh=BIN_THRESH))
#                     per_sample_metrics[w]['weighted_ap_per_sample'].append(weighted_average_precision(pred, gt))
#                     per_sample_metrics[w]['coverage_1%'].append(coverage_topk(pred, gt, 0.01))
#                     per_sample_metrics[w]['coverage_5%'].append(coverage_topk(pred, gt, 0.05))
#                     per_sample_metrics[w]['coverage_10%'].append(coverage_topk(pred, gt, 0.10))
#                     hr, pai = hit_rate_and_pai(pred, gt, 0.01, BIN_THRESH)
#                     per_sample_metrics[w]['hit_1%'].append(hr); per_sample_metrics[w]['pai_1%'].append(pai)
#                     hr, pai = hit_rate_and_pai(pred, gt, 0.05, BIN_THRESH)
#                     per_sample_metrics[w]['hit_5%'].append(hr); per_sample_metrics[w]['pai_5%'].append(pai)
#                     hr, pai = hit_rate_and_pai(pred, gt, 0.10, BIN_THRESH)
#                     per_sample_metrics[w]['hit_10%'].append(hr); per_sample_metrics[w]['pai_10%'].append(pai)

#                     # quantile AP per-sample for each quantile
#                     for q in QUANTILES:
#                         ap_q = quantile_ap_per_sample(pred, gt, q)
#                         quantile_results_per_sample_mean[w][q].append(ap_q)

#             if (batch_idx + 1) % 10 == 0:
#                 print(f"Processed {batch_idx+1}/{len(dataloader)} batches")

#     # ----- 汇总输出 -----
#     def mean_or_nan(lst):
#         arr = np.array([x for x in lst if not (x is None or (isinstance(x, float) and np.isnan(x)))])
#         return float(arr.mean()) if arr.size > 0 else float('nan')

#     summary_rows = []  # 用于写 csv

#     for w in WS_TEST:
#         print(f"\n=== Per-sample mean metrics for guide_w={w} ===")
#         print("RMSE (mean):", mean_or_nan(per_sample_metrics[w]['rmse']))
#         print("Brier (mean):", mean_or_nan(per_sample_metrics[w]['brier']))
#         print("Traditional AP (mean):", mean_or_nan(per_sample_metrics[w]['trad_ap']))
#         print("Relaxed AP r=1 (mean):", mean_or_nan(per_sample_metrics[w]['relaxed_ap_r1']))
#         print("Weighted AP per-sample (mean):", mean_or_nan(per_sample_metrics[w]['weighted_ap_per_sample']))
#         print("Coverage@1% (mean):", mean_or_nan(per_sample_metrics[w]['coverage_1%']))
#         print("Coverage@5% (mean):", mean_or_nan(per_sample_metrics[w]['coverage_5%']))
#         print("Coverage@10% (mean):", mean_or_nan(per_sample_metrics[w]['coverage_10%']))
#         print("Hit@1% (mean):", mean_or_nan(per_sample_metrics[w]['hit_1%']), "PAI@1% (mean):", mean_or_nan(per_sample_metrics[w]['pai_1%']))
#         print("Hit@5% (mean):", mean_or_nan(per_sample_metrics[w]['hit_5%']), "PAI@5% (mean):", mean_or_nan(per_sample_metrics[w]['pai_5%']))
#         print("Hit@10% (mean):", mean_or_nan(per_sample_metrics[w]['hit_10%']), "PAI@10% (mean):", mean_or_nan(per_sample_metrics[w]['pai_10%']))

#         # compute per-sample mean quantile AP for each q
#         per_sample_quantile_mean_list = []
#         for q in QUANTILES:
#             arr = np.array([x for x in quantile_results_per_sample_mean[w][q] if not (x is None or (isinstance(x, float) and np.isnan(x)))])
#             per_sample_mean = float(arr.mean()) if arr.size > 0 else float('nan')
#             per_sample_quantile_mean_list.append(per_sample_mean)
#             print(f"Quantile {q:.2f} per-sample mean AP: {per_sample_mean:.6f}")

#         # global aggregated metrics
#         gm = None
#         try:
#             gm = None
#             # compute global aggregated metrics using all_preds[w], all_gts[w]
#             preds_all = np.concatenate([p.flatten() for p in all_preds[w]])
#             gts_all = np.concatenate([g.flatten() for g in all_gts[w]])
#             mask = np.isfinite(preds_all) & np.isfinite(gts_all)
#             preds_all = preds_all[mask]; gts_all = gts_all[mask]
#             preds_all = np.clip(preds_all, 0.0, None)

#             if preds_all.size > 0:
#                 rmse_global = float(np.sqrt(np.mean((preds_all - gts_all)**2)))
#                 brier_global = float(np.mean((preds_all - gts_all)**2))
#                 # traditional AP (global)
#                 gtbin_global = (gts_all > BIN_THRESH).astype(int)
#                 trad_ap_global = float(average_precision_score(gtbin_global, preds_all)) if gtbin_global.sum() > 0 else float('nan')
#                 # weighted AP global
#                 total_rel = gts_all.sum()
#                 if total_rel == 0:
#                     weighted_ap_global = float('nan')
#                 else:
#                     idx = np.argsort(preds_all)[::-1]
#                     rel_sorted = gts_all[idx]
#                     cumsum_rel = np.cumsum(rel_sorted)
#                     ranks = np.arange(1, len(rel_sorted) + 1)
#                     prec_at_r = cumsum_rel / ranks
#                     weighted_ap_global = float((prec_at_r * rel_sorted).sum() / total_rel)
#                 # coverage top%
#                 covs = {p: coverage_topk(preds_all.reshape(-1,1)[:,0], gts_all.reshape(-1,1)[:,0], p) for p in TOP_PERCENTS}
#                 # hit/PAI global
#                 hits_pais = {}
#                 for p in TOP_PERCENTS:
#                     # left reuse helper: but these helpers accept maps; we operate on flattened arrays directly here
#                     N = len(preds_all)
#                     k = max(1, int(N * p))
#                     idx_top = np.argsort(preds_all)[::-1][:k]
#                     gtbin = (gts_all > BIN_THRESH).astype(int)
#                     if gtbin.sum() == 0:
#                         hits_pais[p] = (float('nan'), float('nan'))
#                     else:
#                         c_pred = gtbin[idx_top].sum()
#                         hit = float(c_pred) / float(gtbin.sum())
#                         area_ratio = float(k) / float(N)
#                         pai = (float(c_pred) / float(gtbin.sum())) / area_ratio if area_ratio > 0 else float('nan')
#                         hits_pais[p] = (hit, pai)

#                 # quantile AP global for each quantile
#                 quantile_ap_global_map = {}
#                 for q in QUANTILES:
#                     if np.all(gts_all == gts_all[0]):
#                         if gts_all[0] == 0:
#                             quantile_ap_global_map[q] = float('nan')
#                         else:
#                             quantile_ap_global_map[q] = float(average_precision_score(np.ones_like(gts_all, dtype=int), preds_all))
#                     else:
#                         thresh = np.quantile(gts_all, q)
#                         y_true = (gts_all >= thresh).astype(int)
#                         if y_true.sum() == 0:
#                             quantile_ap_global_map[q] = float('nan')
#                         else:
#                             quantile_ap_global_map[q] = float(average_precision_score(y_true, preds_all))

#                 # assemble global dict
#                 gm = {
#                     'rmse_global': rmse_global,
#                     'brier_global': brier_global,
#                     'trad_ap_global': trad_ap_global,
#                     'weighted_ap_global': weighted_ap_global,
#                     'coverage': covs,
#                     'hits_pais': hits_pais,
#                     'quantile_ap_global': quantile_ap_global_map,
#                     'preds_all_size': preds_all.size
#                 }
#             else:
#                 gm = None
#         except Exception as e:
#             print("Error computing global metrics:", e)
#             gm = None

#         print(f"\n=== Global aggregated metrics for guide_w={w} ===")
#         print(gm)

#         # 保存 summary 行到 csv（per-sample mean + global）
#         row = {
#             'guide_w': w,
#             'per_rmse_mean': mean_or_nan(per_sample_metrics[w]['rmse']),
#             'per_brier_mean': mean_or_nan(per_sample_metrics[w]['brier']),
#             'per_trad_ap_mean': mean_or_nan(per_sample_metrics[w]['trad_ap']),
#             'per_relaxed_ap_r1_mean': mean_or_nan(per_sample_metrics[w]['relaxed_ap_r1']),
#             'per_weighted_ap_mean': mean_or_nan(per_sample_metrics[w]['weighted_ap_per_sample']),
#             'per_cov1_mean': mean_or_nan(per_sample_metrics[w]['coverage_1%']),
#             'per_cov5_mean': mean_or_nan(per_sample_metrics[w]['coverage_5%']),
#             'per_cov10_mean': mean_or_nan(per_sample_metrics[w]['coverage_10%']),
#         }
#         # add global if exist
#         if gm is not None:
#             row.update({
#                 'global_rmse': gm['rmse_global'],
#                 'global_brier': gm['brier_global'],
#                 'global_trad_ap': gm['trad_ap_global'],
#                 'global_weighted_ap': gm['weighted_ap_global'],
#                 'global_cov1': gm['coverage'].get(0.01, float('nan')),
#                 'global_cov5': gm['coverage'].get(0.05, float('nan')),
#                 'global_cov10': gm['coverage'].get(0.10, float('nan')),
#                 'global_preds_all_size': gm['preds_all_size']
#             })
#         summary_rows.append(row)

#         # 画图：quantile vs AP（per-sample mean & global）
#         per_means = per_sample_quantile_mean_list
#         global_vals = [gm['quantile_ap_global'].get(q, float('nan')) if gm is not None else float('nan') for q in QUANTILES]
#         plt.figure(figsize=(8,5))
#         plt.plot(QUANTILES, per_means, marker='o', label='Per-sample mean AP')
#         plt.plot(QUANTILES, global_vals, marker='s', label='Global aggregated AP')
#         plt.xlabel('Quantile (threshold percentile of GT)')
#         plt.ylabel('Average Precision (AUPRC)')
#         plt.title(f'Quantile sensitivity (guide_w={w})')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         figpath = os.path.join(SAVE_DIR, f'quantile_sensitivity_w{w}.png')
#         plt.savefig(figpath, dpi=200)
#         plt.close()
#         print(f"Saved quantile sensitivity plot to {figpath}")

#         # 画 coverage 曲线（global preds/gts 必须存在）
#         if gm is not None:
#             preds_arr = preds_all
#             gts_arr = gts_all
#             coverages = []
#             for tp in COVERAGE_TOP_PCT_RANGE:
#                 N = len(preds_arr)
#                 k = max(1, int(N * tp))
#                 idx_top = np.argsort(preds_arr)[::-1][:k]
#                 if gts_arr.sum() == 0:
#                     coverages.append(float('nan'))
#                 else:
#                     coverages.append(float(gts_arr[idx_top].sum() / gts_arr.sum()))
#             plt.figure(figsize=(8,5))
#             plt.plot(COVERAGE_TOP_PCT_RANGE*100, coverages, label='Coverage vs Top% Area')
#             plt.xlabel('Top area percent (%)')
#             plt.ylabel('Coverage (fraction of total risk)')
#             plt.title(f'Coverage curve (guide_w={w})')
#             plt.grid(True)
#             plt.tight_layout()
#             figcov = os.path.join(SAVE_DIR, f'coverage_curve_w{w}.png')
#             plt.savefig(figcov, dpi=200)
#             plt.close()
#             print(f"Saved coverage curve to {figcov}")

#     # 写 csv summary
#     csv_path = os.path.join(SAVE_DIR, "evaluation_summary.csv")
#     keys = sorted(summary_rows[0].keys()) if summary_rows else []
#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         for r in summary_rows:
#             writer.writerow(r)
#     print(f"Saved summary CSV to {csv_path}")

#     print("All done. Results and plots saved in", SAVE_DIR)


# if __name__ == "__main__":
#     main()

# eval_quantile_sensitivity.py
# """
# 完整评估脚本：
# - 将预测负值裁为 0
# - 计算并输出：RMSE/Brier、Traditional AP (bin=0.1)、Weighted AP（global & per-sample mean）
#   Quantile-threshold AP 敏感性分析（quantile 0.3~0.9 step 0.05）
#   Coverage@top% / HitRate@PAI@top%（top1%,5%,10%）
# - 输出 per-sample mean 与 global aggregated 两种结果
# - 绘图并保存到 ./Result/Eval/
# - [新增] 保存预测结果 (.npy) 到 ./Result/Predictions/
# """
# import os
# import math
# import csv
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch.utils.data import DataLoader
# from sklearn.metrics import average_precision_score
# from scipy.ndimage import binary_dilation

# from RiskmapDataset import RiskMapTest
# from DiffResCrime import STFusionNet, DiffRes

# # ----- 配置 -----
# CHECKPOINT_PATH = "./Result/Output/NY.pth"
# # CHECKPOINT_PATH = "./Result/Output/best_network_with_early_stopping.pth"

# # 评估指标保存目录
# SAVE_DIR = "./Result/Eval/All"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # 预测结果保存目录 (新增)
# PRED_SAVE_ROOT = "./Result/Predictions"
# os.makedirs(PRED_SAVE_ROOT, exist_ok=True)

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 16
# N_FEAT = 256
# N_T = 15

# # WS_TEST = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]    # list of guide_w to evaluate
# # WS_TEST = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
# WS_TEST = [0.0]
# CLIP_NEGATIVE = False
# SAVE_PREDICTIONS = True  # 开关：是否保存 .npy 文件
# # [新增] 硬阈值：小于此值的预测一律视为 0
# # 根据你的直方图，噪音主要集中在 10^-2 以下，建议尝试 1e-3 或 1e-2
# HARD_THRESHOLD = 2.5e-4

# # quantile sensitivity range (0.3 -> 0.9 inclusive, step 0.05)
# QUANTILES = np.arange(0.3, 0.91, 0.05)

# # top% list for coverage/hit/PAI
# TOP_PERCENTS = [0.01, 0.05, 0.10]

# # fixed bin threshold for "traditional" AP and HitRate/PAI
# # BIN_THRESH = 0.1
# BIN_THRESH = 0.0

# # coverage curve resolution (for plotting)
# COVERAGE_TOP_PCT_RANGE = np.linspace(0.005, 0.20, 80)  # 0.5% ~ 20%

# # ----------------- 工具函数 -----------------
# def clamp_neg_to_zero(pred):
#     return np.clip(np.array(pred, dtype=float), 0.0, None)

# def rmse_map(pred_map, gt_map):
#     mask = np.isfinite(pred_map) & np.isfinite(gt_map)
#     if not mask.any(): return float('nan')
#     return float(np.sqrt(np.mean((pred_map[mask] - gt_map[mask])**2)))

# def brier_map(pred_map, gt_map):
#     mask = np.isfinite(pred_map) & np.isfinite(gt_map)
#     if not mask.any(): return float('nan')
#     return float(np.mean((pred_map[mask] - gt_map[mask])**2))

# def traditional_ap(pred_map, gt_map, bin_thresh=BIN_THRESH):
#     pred = pred_map.flatten()
#     gt_bin = (gt_map.flatten() > bin_thresh).astype(int)
#     if gt_bin.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(gt_bin, pred))

# def relaxed_ap(pred_map, gt_map, dilation_radius=1, bin_thresh=BIN_THRESH):
#     gt_bin = (gt_map > bin_thresh).astype(np.uint8)
#     structure = np.ones((2*dilation_radius+1, 2*dilation_radius+1), dtype=np.uint8)
#     gt_dilated = binary_dilation(gt_bin, structure=structure).astype(int)
#     if gt_dilated.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(gt_dilated.flatten().astype(int), pred_map.flatten()))

# def weighted_average_precision(pred_map, label_map):
#     pred = pred_map.flatten()
#     rel = label_map.flatten().astype(float)
#     total_rel = rel.sum()
#     if total_rel == 0:
#         return float('nan')
#     idx = np.argsort(pred)[::-1]
#     rel_sorted = rel[idx]
#     cumsum_rel = np.cumsum(rel_sorted)
#     ranks = np.arange(1, len(rel_sorted) + 1)
#     precision_at_r = cumsum_rel / ranks
#     wap = (precision_at_r * rel_sorted).sum() / total_rel
#     return float(wap)

# def coverage_topk(pred_map, label_map, top_percent):
#     pred = pred_map.flatten()
#     rel = label_map.flatten().astype(float)
#     N = len(pred)
#     k = max(1, int(N * top_percent))
#     idx = np.argsort(pred)[::-1][:k]
#     total_rel = rel.sum()
#     if total_rel == 0:
#         return float('nan')
#     return float(rel[idx].sum() / total_rel)

# def hit_rate_and_pai(pred_map, gt_map, top_percent, bin_thresh=BIN_THRESH):
#     pred = pred_map.flatten()
#     gt = gt_map.flatten()
#     gt_bin = (gt > bin_thresh).astype(int)
#     N = len(pred)
#     k = max(1, int(N * top_percent))
#     idx = np.argsort(pred)[::-1][:k]
#     c_total = gt_bin.sum()
#     if c_total == 0:
#         return float('nan'), float('nan')
#     c_pred = gt_bin[idx].sum()
#     hit = float(c_pred) / float(c_total)
#     area_ratio = float(k) / float(N)
#     pai = (float(c_pred) / float(c_total)) / area_ratio if area_ratio > 0 else float('nan')
#     return hit, pai

# # quantile-threshold AP (per-sample)
# def quantile_ap_per_sample(pred_map, gt_map, quantile):
#     """
#     在单个样本级别：以该样本的 gt_map 的 quantile 作阈值，二值化 gt 后计算 AP
#     返回 AP（或 nan）
#     """
#     gt_flat = gt_map.flatten()
#     if np.all(gt_flat == gt_flat[0]):
#         if gt_flat[0] == 0:
#             return float('nan')
#         else:
#             # 全正样本 -> AP 可以计算为 average_precision_score(all 1s, pred)
#             y_true = np.ones_like(gt_flat, dtype=int)
#             return float(average_precision_score(y_true, pred_map.flatten()))
#     threshold = np.quantile(gt_flat, quantile)
#     y_true = (gt_flat >= threshold).astype(int)
#     if y_true.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(y_true, pred_map.flatten()))

# # quantile-threshold AP (global: 所有样本合并后用全局 gt quantile 作阈值)
# def quantile_ap_global(pred_list, gt_list, quantile):
#     preds = np.concatenate([p.flatten() for p in pred_list])
#     gts = np.concatenate([g.flatten() for g in gt_list])
#     mask = np.isfinite(preds) & np.isfinite(gts)
#     preds = preds[mask]; gts = gts[mask]
#     if preds.size == 0:
#         return float('nan')
#     if np.all(gts == gts[0]):
#         if gts[0] == 0:
#             return float('nan')
#         else:
#             return float(average_precision_score(np.ones_like(gts, dtype=int), preds))
#     thresh = np.quantile(gts, quantile)
#     y_true = (gts >= thresh).astype(int)
#     if y_true.sum() == 0:
#         return float('nan')
#     return float(average_precision_score(y_true, preds))


# # ----------------- 主流程 -----------------
# def main():
#     # load model
#     # fusion_net = STFusionNet(in_channels=1, n_feat=N_FEAT, poi_channels=13)
#     fusion_net = STFusionNet(
#     in_channels=1,
#     n_feat=N_FEAT,        # 256
#     poi_channels=13,
#     x_dim=2048            # 1024
#     )
#     # diffres_model = DiffRes(backbone=fusion_net, n_T=N_T, device=DEVICE, drop_prob=0.1)
#     diffres_model = DiffRes(
#         backbone=fusion_net, 
#         n_T=N_T, 
#         device=DEVICE, 
#         drop_prob=0.1, 
#         # gamma=0.1,
#         gamma=0.0,
#         power=2.0
#     )

#     if not os.path.exists(CHECKPOINT_PATH):
#         raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
#     diffres_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
#     diffres_model.to(DEVICE)
#     diffres_model.eval()
#     print("Model loaded ->", CHECKPOINT_PATH)
#     print("Device:", DEVICE)

#     # dataloader
#     dataset = RiskMapTest()
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=False)
#     print("Dataset size:", len(dataset), "batches:", len(dataloader))

#     # containers
#     per_sample_metrics = {w: {'rmse': [], 'brier': [], 'trad_ap': [], 'relaxed_ap_r1': [], 'weighted_ap_per_sample': [],
#                              'coverage_1%': [], 'coverage_5%': [], 'coverage_10%': [],
#                              'hit_1%': [], 'pai_1%': [], 'hit_5%': [], 'pai_5%': [], 'hit_10%': [], 'pai_10%': []}
#                           for w in WS_TEST}
#     all_preds = {w: [] for w in WS_TEST}
#     all_gts = {w: [] for w in WS_TEST}

#     # 记录 quantile AP across samples (per-sample mean) for each w and quantile
#     quantile_results_per_sample_mean = {w: {q: [] for q in QUANTILES} for w in WS_TEST}

#     # Prepare save directories for each w if needed
#     if SAVE_PREDICTIONS:
#         for w in WS_TEST:
#             w_dir = os.path.join(PRED_SAVE_ROOT, f"w_{w}")
#             os.makedirs(w_dir, exist_ok=True)
#             print(f"Predictions for w={w} will be saved to: {w_dir}")

#     # evaluation loop
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(dataloader):
#             filenames = batch['file_name']            # list of strings
#             labels = batch['riskmap_obs']             # Tensor (B,1,H,W)
#             cond_map = batch['map']
#             cond_satellite = batch['satellite']
#             history = batch['riskmap_his']
#             poi = batch['poi']
 
#             # 合并卫星和地图条件 [B, H, W, 6]
#             cond_combined = torch.cat([cond_satellite, cond_map], dim=-1)

#             labels = labels.to(DEVICE)
#             cond_combined = cond_combined.to(DEVICE)
#             history = history.to(DEVICE)
#             poi = poi.to(DEVICE)

#             B = labels.shape[0]
#             for w in WS_TEST:
#                 try:
#                     x_gen = diffres_model.sample(n_sample=B, size=(1, 16, 16),
#                                         cond=cond_combined, x_hist=history, poi=poi, guide_w=w)
#                 except Exception as e:
#                     print(f"Sampling error in batch {batch_idx} w={w}: {e}")
#                     continue

#                 if isinstance(x_gen, torch.Tensor):
#                     x_gen = x_gen.cpu().numpy()

#                 # Current batch save path
#                 current_save_dir = os.path.join(PRED_SAVE_ROOT, f"w_{w}")

#                 for i in range(B):
#                     pred = x_gen[i, 0].astype(float)
#                     gt = labels[i, 0].cpu().numpy().astype(float)
#                     fn = filenames[i] # 获取文件名

#                     if CLIP_NEGATIVE:
#                         pred = clamp_neg_to_zero(pred)

#                     # -------------------------------------------------------
#                     # [新增步骤] 2. 硬阈值过滤 (Hard Thresholding)
#                     # -------------------------------------------------------
#                     # 逻辑：如果预测值非常小（且 GT 极大概率是 0），这通常是扩散模型的背景底噪
#                     # 这一步能消除散点图(Scatter Plot)中竖轴(X=0)上的那些杂点
#                     pred[pred < HARD_THRESHOLD] = 0.0
#                     # -------------------------------------------------------

#                     # --- [NEW] Save Prediction ---
#                     if SAVE_PREDICTIONS:
#                         save_path = os.path.join(current_save_dir, f"{fn}.npy")
#                         np.save(save_path, pred)
#                     # -----------------------------

#                     # store for global metrics
#                     all_preds[w].append(pred.copy())
#                     all_gts[w].append(gt.copy())

#                     # per-sample metrics
#                     per_sample_metrics[w]['rmse'].append(rmse_map(pred, gt))
#                     per_sample_metrics[w]['brier'].append(brier_map(pred, gt))
#                     per_sample_metrics[w]['trad_ap'].append(traditional_ap(pred, gt, BIN_THRESH))
#                     per_sample_metrics[w]['relaxed_ap_r1'].append(relaxed_ap(pred, gt, dilation_radius=1, bin_thresh=BIN_THRESH))
#                     per_sample_metrics[w]['weighted_ap_per_sample'].append(weighted_average_precision(pred, gt))
#                     per_sample_metrics[w]['coverage_1%'].append(coverage_topk(pred, gt, 0.01))
#                     per_sample_metrics[w]['coverage_5%'].append(coverage_topk(pred, gt, 0.05))
#                     per_sample_metrics[w]['coverage_10%'].append(coverage_topk(pred, gt, 0.10))
#                     hr, pai = hit_rate_and_pai(pred, gt, 0.01, BIN_THRESH)
#                     per_sample_metrics[w]['hit_1%'].append(hr); per_sample_metrics[w]['pai_1%'].append(pai)
#                     hr, pai = hit_rate_and_pai(pred, gt, 0.05, BIN_THRESH)
#                     per_sample_metrics[w]['hit_5%'].append(hr); per_sample_metrics[w]['pai_5%'].append(pai)
#                     hr, pai = hit_rate_and_pai(pred, gt, 0.10, BIN_THRESH)
#                     per_sample_metrics[w]['hit_10%'].append(hr); per_sample_metrics[w]['pai_10%'].append(pai)

#                     # quantile AP per-sample for each quantile
#                     for q in QUANTILES:
#                         ap_q = quantile_ap_per_sample(pred, gt, q)
#                         quantile_results_per_sample_mean[w][q].append(ap_q)

#             if (batch_idx + 1) % 10 == 0:
#                 print(f"Processed {batch_idx+1}/{len(dataloader)} batches")

#     # ----- 汇总输出 -----
#     def mean_or_nan(lst):
#         arr = np.array([x for x in lst if not (x is None or (isinstance(x, float) and np.isnan(x)))])
#         return float(arr.mean()) if arr.size > 0 else float('nan')

#     summary_rows = []  # 用于写 csv

#     for w in WS_TEST:
#         print(f"\n=== Per-sample mean metrics for guide_w={w} ===")
#         print("RMSE (mean):", mean_or_nan(per_sample_metrics[w]['rmse']))
#         print("Brier (mean):", mean_or_nan(per_sample_metrics[w]['brier']))
#         print("Traditional AP (mean):", mean_or_nan(per_sample_metrics[w]['trad_ap']))
#         print("Relaxed AP r=1 (mean):", mean_or_nan(per_sample_metrics[w]['relaxed_ap_r1']))
#         print("Weighted AP per-sample (mean):", mean_or_nan(per_sample_metrics[w]['weighted_ap_per_sample']))
#         print("Coverage@1% (mean):", mean_or_nan(per_sample_metrics[w]['coverage_1%']))
#         print("Coverage@5% (mean):", mean_or_nan(per_sample_metrics[w]['coverage_5%']))
#         print("Coverage@10% (mean):", mean_or_nan(per_sample_metrics[w]['coverage_10%']))
#         print("Hit@1% (mean):", mean_or_nan(per_sample_metrics[w]['hit_1%']), "PAI@1% (mean):", mean_or_nan(per_sample_metrics[w]['pai_1%']))
#         print("Hit@5% (mean):", mean_or_nan(per_sample_metrics[w]['hit_5%']), "PAI@5% (mean):", mean_or_nan(per_sample_metrics[w]['pai_5%']))
#         print("Hit@10% (mean):", mean_or_nan(per_sample_metrics[w]['hit_10%']), "PAI@10% (mean):", mean_or_nan(per_sample_metrics[w]['pai_10%']))

#         # compute per-sample mean quantile AP for each q
#         per_sample_quantile_mean_list = []
#         for q in QUANTILES:
#             arr = np.array([x for x in quantile_results_per_sample_mean[w][q] if not (x is None or (isinstance(x, float) and np.isnan(x)))])
#             per_sample_mean = float(arr.mean()) if arr.size > 0 else float('nan')
#             per_sample_quantile_mean_list.append(per_sample_mean)
#             print(f"Quantile {q:.2f} per-sample mean AP: {per_sample_mean:.6f}")

#         # global aggregated metrics
#         gm = None
#         try:
#             gm = None
#             # compute global aggregated metrics using all_preds[w], all_gts[w]
#             preds_all = np.concatenate([p.flatten() for p in all_preds[w]])
#             gts_all = np.concatenate([g.flatten() for g in all_gts[w]])
#             mask = np.isfinite(preds_all) & np.isfinite(gts_all)
#             preds_all = preds_all[mask]; gts_all = gts_all[mask]
#             preds_all = np.clip(preds_all, 0.0, None)

#             if preds_all.size > 0:
#                 rmse_global = float(np.sqrt(np.mean((preds_all - gts_all)**2)))
#                 brier_global = float(np.mean((preds_all - gts_all)**2))
#                 # traditional AP (global)
#                 gtbin_global = (gts_all > BIN_THRESH).astype(int)
#                 trad_ap_global = float(average_precision_score(gtbin_global, preds_all)) if gtbin_global.sum() > 0 else float('nan')
#                 # weighted AP global
#                 total_rel = gts_all.sum()
#                 if total_rel == 0:
#                     weighted_ap_global = float('nan')
#                 else:
#                     idx = np.argsort(preds_all)[::-1]
#                     rel_sorted = gts_all[idx]
#                     cumsum_rel = np.cumsum(rel_sorted)
#                     ranks = np.arange(1, len(rel_sorted) + 1)
#                     prec_at_r = cumsum_rel / ranks
#                     weighted_ap_global = float((prec_at_r * rel_sorted).sum() / total_rel)
#                 # coverage top%
#                 covs = {p: coverage_topk(preds_all.reshape(-1,1)[:,0], gts_all.reshape(-1,1)[:,0], p) for p in TOP_PERCENTS}
#                 # hit/PAI global
#                 hits_pais = {}
#                 for p in TOP_PERCENTS:
#                     # left reuse helper: but these helpers accept maps; we operate on flattened arrays directly here
#                     N = len(preds_all)
#                     k = max(1, int(N * p))
#                     idx_top = np.argsort(preds_all)[::-1][:k]
#                     gtbin = (gts_all > BIN_THRESH).astype(int)
#                     if gtbin.sum() == 0:
#                         hits_pais[p] = (float('nan'), float('nan'))
#                     else:
#                         c_pred = gtbin[idx_top].sum()
#                         hit = float(c_pred) / float(gtbin.sum())
#                         area_ratio = float(k) / float(N)
#                         pai = (float(c_pred) / float(gtbin.sum())) / area_ratio if area_ratio > 0 else float('nan')
#                         hits_pais[p] = (hit, pai)

#                 # quantile AP global for each quantile
#                 quantile_ap_global_map = {}
#                 for q in QUANTILES:
#                     if np.all(gts_all == gts_all[0]):
#                         if gts_all[0] == 0:
#                             quantile_ap_global_map[q] = float('nan')
#                         else:
#                             quantile_ap_global_map[q] = float(average_precision_score(np.ones_like(gts_all, dtype=int), preds_all))
#                     else:
#                         thresh = np.quantile(gts_all, q)
#                         y_true = (gts_all >= thresh).astype(int)
#                         if y_true.sum() == 0:
#                             quantile_ap_global_map[q] = float('nan')
#                         else:
#                             quantile_ap_global_map[q] = float(average_precision_score(y_true, preds_all))

#                 # assemble global dict
#                 gm = {
#                     'rmse_global': rmse_global,
#                     'brier_global': brier_global,
#                     'trad_ap_global': trad_ap_global,
#                     'weighted_ap_global': weighted_ap_global,
#                     'coverage': covs,
#                     'hits_pais': hits_pais,
#                     'quantile_ap_global': quantile_ap_global_map,
#                     'preds_all_size': preds_all.size
#                 }
#             else:
#                 gm = None
#         except Exception as e:
#             print("Error computing global metrics:", e)
#             gm = None

#         print(f"\n=== Global aggregated metrics for guide_w={w} ===")
#         print(gm)

#         # 保存 summary 行到 csv（per-sample mean + global）
#         row = {
#             'guide_w': w,
#             'per_rmse_mean': mean_or_nan(per_sample_metrics[w]['rmse']),
#             'per_brier_mean': mean_or_nan(per_sample_metrics[w]['brier']),
#             'per_trad_ap_mean': mean_or_nan(per_sample_metrics[w]['trad_ap']),
#             'per_relaxed_ap_r1_mean': mean_or_nan(per_sample_metrics[w]['relaxed_ap_r1']),
#             'per_weighted_ap_mean': mean_or_nan(per_sample_metrics[w]['weighted_ap_per_sample']),
#             'per_cov1_mean': mean_or_nan(per_sample_metrics[w]['coverage_1%']),
#             'per_cov5_mean': mean_or_nan(per_sample_metrics[w]['coverage_5%']),
#             'per_cov10_mean': mean_or_nan(per_sample_metrics[w]['coverage_10%']),
#         }
#         # add global if exist
#         if gm is not None:
#             row.update({
#                 'global_rmse': gm['rmse_global'],
#                 'global_brier': gm['brier_global'],
#                 'global_trad_ap': gm['trad_ap_global'],
#                 'global_weighted_ap': gm['weighted_ap_global'],
#                 'global_cov1': gm['coverage'].get(0.01, float('nan')),
#                 'global_cov5': gm['coverage'].get(0.05, float('nan')),
#                 'global_cov10': gm['coverage'].get(0.10, float('nan')),
#                 'global_preds_all_size': gm['preds_all_size']
#             })
#         summary_rows.append(row)

#         # 画图：quantile vs AP（per-sample mean & global）
#         per_means = per_sample_quantile_mean_list
#         global_vals = [gm['quantile_ap_global'].get(q, float('nan')) if gm is not None else float('nan') for q in QUANTILES]
#         plt.figure(figsize=(8,5))
#         plt.plot(QUANTILES, per_means, marker='o', label='Per-sample mean AP')
#         plt.plot(QUANTILES, global_vals, marker='s', label='Global aggregated AP')
#         plt.xlabel('Quantile (threshold percentile of GT)')
#         plt.ylabel('Average Precision (AUPRC)')
#         plt.title(f'Quantile sensitivity (guide_w={w})')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         figpath = os.path.join(SAVE_DIR, f'quantile_sensitivity_w{w}.png')
#         plt.savefig(figpath, dpi=200)
#         plt.close()
#         print(f"Saved quantile sensitivity plot to {figpath}")

#         # 画 coverage 曲线（global preds/gts 必须存在）
#         if gm is not None:
#             preds_arr = preds_all
#             gts_arr = gts_all
#             coverages = []
#             for tp in COVERAGE_TOP_PCT_RANGE:
#                 N = len(preds_arr)
#                 k = max(1, int(N * tp))
#                 idx_top = np.argsort(preds_arr)[::-1][:k]
#                 if gts_arr.sum() == 0:
#                     coverages.append(float('nan'))
#                 else:
#                     coverages.append(float(gts_arr[idx_top].sum() / gts_arr.sum()))
#             plt.figure(figsize=(8,5))
#             plt.plot(COVERAGE_TOP_PCT_RANGE*100, coverages, label='Coverage vs Top% Area')
#             plt.xlabel('Top area percent (%)')
#             plt.ylabel('Coverage (fraction of total risk)')
#             plt.title(f'Coverage curve (guide_w={w})')
#             plt.grid(True)
#             plt.tight_layout()
#             figcov = os.path.join(SAVE_DIR, f'coverage_curve_w{w}.png')
#             plt.savefig(figcov, dpi=200)
#             plt.close()
#             print(f"Saved coverage curve to {figcov}")

#     # 写 csv summary
#     csv_path = os.path.join(SAVE_DIR, "evaluation_summary.csv")
#     keys = sorted(summary_rows[0].keys()) if summary_rows else []
#     with open(csv_path, 'w', newline='') as f:
#         writer = csv.DictWriter(f, fieldnames=keys)
#         writer.writeheader()
#         for r in summary_rows:
#             writer.writerow(r)
#     print(f"Saved summary CSV to {csv_path}")

#     print("All done. Results and plots saved in", SAVE_DIR)
#     if SAVE_PREDICTIONS:
#         print("Prediction .npy files saved in", PRED_SAVE_ROOT)


# if __name__ == "__main__":
#     main()

"""
完整模型 (Baseline) 评估脚本：
- 使用 DiffResCrime.py 中的原始 STFusionNet (含 CondSEBlock + Attention Fusion)。
- 输入完整 POI 数据。
- 评价指标与消融实验脚本完全对齐。
"""
import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from scipy.ndimage import binary_dilation

from RiskmapDataset import RiskMapTest
# ================= [关键点] =================
# 从原始 DiffResCrime 导入完整模型
from DiffResCrime import STFusionNet, DiffRes
# ===========================================

# ----- 配置 -----
# 对应 train.py 保存的完整模型权重
CHECKPOINT_PATH = "./Result/Output/NY.pth"
SAVE_DIR = "./Result/Eval/Full_Model"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
N_FEAT = 256
N_T = 15
WS_TEST = [0.0]  # 仅测试 guide_w=0，保持与消融实验一致
CLIP_NEGATIVE = True

# 评价指标参数
QUANTILES = np.arange(0.3, 0.91, 0.05)
TOP_PERCENTS = [0.01, 0.05, 0.10]
BIN_THRESH = 0.0
COVERAGE_TOP_PCT_RANGE = np.linspace(0.005, 0.20, 80)

# ----------------- 工具函数 (保持完全一致) -----------------
def clamp_neg_to_zero(pred):
    return np.clip(np.array(pred, dtype=float), 0.0, None)

def rmse_map(pred_map, gt_map):
    mask = np.isfinite(pred_map) & np.isfinite(gt_map)
    if not mask.any(): return float('nan')
    return float(np.sqrt(np.mean((pred_map[mask] - gt_map[mask])**2)))

def brier_map(pred_map, gt_map):
    mask = np.isfinite(pred_map) & np.isfinite(gt_map)
    if not mask.any(): return float('nan')
    return float(np.mean((pred_map[mask] - gt_map[mask])**2))

def traditional_ap(pred_map, gt_map, bin_thresh=BIN_THRESH):
    pred = pred_map.flatten()
    gt_bin = (gt_map.flatten() > bin_thresh).astype(int)
    if gt_bin.sum() == 0: return float('nan')
    return float(average_precision_score(gt_bin, pred))

def relaxed_ap(pred_map, gt_map, dilation_radius=1, bin_thresh=BIN_THRESH):
    gt_bin = (gt_map > bin_thresh).astype(np.uint8)
    structure = np.ones((2*dilation_radius+1, 2*dilation_radius+1), dtype=np.uint8)
    gt_dilated = binary_dilation(gt_bin, structure=structure).astype(int)
    if gt_dilated.sum() == 0: return float('nan')
    return float(average_precision_score(gt_dilated.flatten().astype(int), pred_map.flatten()))

def weighted_average_precision(pred_map, label_map):
    pred = pred_map.flatten()
    rel = label_map.flatten().astype(float)
    total_rel = rel.sum()
    if total_rel == 0: return float('nan')
    idx = np.argsort(pred)[::-1]
    rel_sorted = rel[idx]
    cumsum_rel = np.cumsum(rel_sorted)
    ranks = np.arange(1, len(rel_sorted) + 1)
    precision_at_r = cumsum_rel / ranks
    wap = (precision_at_r * rel_sorted).sum() / total_rel
    return float(wap)

def coverage_topk(pred_map, label_map, top_percent):
    pred = pred_map.flatten()
    rel = label_map.flatten().astype(float)
    N = len(pred)
    k = max(1, int(N * top_percent))
    idx = np.argsort(pred)[::-1][:k]
    total_rel = rel.sum()
    if total_rel == 0: return float('nan')
    return float(rel[idx].sum() / total_rel)

def hit_rate_and_pai(pred_map, gt_map, top_percent, bin_thresh=BIN_THRESH):
    pred = pred_map.flatten()
    gt = gt_map.flatten()
    gt_bin = (gt > bin_thresh).astype(int)
    N = len(pred)
    k = max(1, int(N * top_percent))
    idx = np.argsort(pred)[::-1][:k]
    c_total = gt_bin.sum()
    if c_total == 0: return float('nan'), float('nan')
    c_pred = gt_bin[idx].sum()
    hit = float(c_pred) / float(c_total)
    area_ratio = float(k) / float(N)
    pai = (float(c_pred) / float(c_total)) / area_ratio if area_ratio > 0 else float('nan')
    return hit, pai

def quantile_ap_per_sample(pred_map, gt_map, quantile):
    gt_flat = gt_map.flatten()
    if np.all(gt_flat == gt_flat[0]):
        if gt_flat[0] == 0: return float('nan')
        else:
            y_true = np.ones_like(gt_flat, dtype=int)
            return float(average_precision_score(y_true, pred_map.flatten()))
    threshold = np.quantile(gt_flat, quantile)
    y_true = (gt_flat >= threshold).astype(int)
    if y_true.sum() == 0: return float('nan')
    return float(average_precision_score(y_true, pred_map.flatten()))

# ----------------- 主流程 -----------------
def main():
    print(f"Starting Full Model (Baseline) Evaluation...")
    print(f"Loading model from: {CHECKPOINT_PATH}")

    # 1. 模型初始化 (Standard STFusionNet)
    fusion_net = STFusionNet(
        in_channels=1,
        n_feat=N_FEAT,
        poi_channels=13,
        x_dim=2048
    )
    # diffres_model = DiffRes(backbone=fusion_net, n_T=N_T, device=DEVICE, drop_prob=0.1)
    diffres_model = DiffRes(
        backbone=fusion_net, 
        n_T=N_T, 
        device=DEVICE, 
        drop_prob=0.1, 
        gamma=0.1,
        # gamma=0.0,
        power=2.0
    )

    # 2. 加载权重
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}. Please run train.py first.")
    
    diffres_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    diffres_model.to(DEVICE)
    diffres_model.eval()

    # 3. 数据集
    dataset = RiskMapTest()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=False)
    print("Dataset size:", len(dataset))

    # 容器初始化
    per_sample_metrics = {w: {'rmse': [], 'brier': [], 'trad_ap': [], 'relaxed_ap_r1': [], 'weighted_ap_per_sample': [],
                             'coverage_1%': [], 'coverage_5%': [], 'coverage_10%': [],
                             'hit_1%': [], 'pai_1%': [], 'hit_5%': [], 'pai_5%': [], 'hit_10%': [], 'pai_10%': []}
                          for w in WS_TEST}
    all_preds = {w: [] for w in WS_TEST}
    all_gts = {w: [] for w in WS_TEST}
    quantile_results_per_sample_mean = {w: {q: [] for q in QUANTILES} for w in WS_TEST}

    # 4. 评估循环
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            labels = batch['riskmap_obs']
            cond_map = batch['map']
            cond_satellite = batch['satellite']
            history = batch['riskmap_his']
            poi = batch['poi']
 
            # 完整模型：正常输入所有模态
            cond_combined = torch.cat([cond_satellite, cond_map], dim=-1)

            labels = labels.to(DEVICE)
            cond_combined = cond_combined.to(DEVICE)
            history = history.to(DEVICE)
            poi = poi.to(DEVICE)

            B = labels.shape[0]
            for w in WS_TEST:
                try:
                    x_gen = diffres_model.sample(n_sample=B, size=(1, 16, 16),
                                        cond=cond_combined, x_hist=history, poi=poi, guide_w=w)
                except Exception as e:
                    print(f"Sampling error in batch {batch_idx} w={w}: {e}")
                    continue

                if isinstance(x_gen, torch.Tensor):
                    x_gen = x_gen.cpu().numpy()

                for i in range(B):
                    pred = x_gen[i, 0].astype(float)
                    gt = labels[i, 0].cpu().numpy().astype(float)

                    if CLIP_NEGATIVE:
                        pred = clamp_neg_to_zero(pred)

                    # Store for Global Aggregation
                    all_preds[w].append(pred.copy())
                    all_gts[w].append(gt.copy())

                    # Per-sample metrics
                    per_sample_metrics[w]['rmse'].append(rmse_map(pred, gt))
                    per_sample_metrics[w]['brier'].append(brier_map(pred, gt))
                    per_sample_metrics[w]['trad_ap'].append(traditional_ap(pred, gt, BIN_THRESH))
                    per_sample_metrics[w]['relaxed_ap_r1'].append(relaxed_ap(pred, gt, dilation_radius=1, bin_thresh=BIN_THRESH))
                    per_sample_metrics[w]['weighted_ap_per_sample'].append(weighted_average_precision(pred, gt))
                    
                    per_sample_metrics[w]['coverage_1%'].append(coverage_topk(pred, gt, 0.01))
                    per_sample_metrics[w]['coverage_5%'].append(coverage_topk(pred, gt, 0.05))
                    per_sample_metrics[w]['coverage_10%'].append(coverage_topk(pred, gt, 0.10))
                    
                    hr, pai = hit_rate_and_pai(pred, gt, 0.01, BIN_THRESH)
                    per_sample_metrics[w]['hit_1%'].append(hr); per_sample_metrics[w]['pai_1%'].append(pai)
                    hr, pai = hit_rate_and_pai(pred, gt, 0.05, BIN_THRESH)
                    per_sample_metrics[w]['hit_5%'].append(hr); per_sample_metrics[w]['pai_5%'].append(pai)
                    hr, pai = hit_rate_and_pai(pred, gt, 0.10, BIN_THRESH)
                    per_sample_metrics[w]['hit_10%'].append(hr); per_sample_metrics[w]['pai_10%'].append(pai)

                    # Quantile AP
                    for q in QUANTILES:
                        ap_q = quantile_ap_per_sample(pred, gt, q)
                        quantile_results_per_sample_mean[w][q].append(ap_q)

            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluated {batch_idx+1}/{len(dataloader)} batches")

    # 5. 汇总与输出
    def mean_or_nan(lst):
        arr = np.array([x for x in lst if not (x is None or (isinstance(x, float) and np.isnan(x)))])
        return float(arr.mean()) if arr.size > 0 else float('nan')

    summary_rows = []

    for w in WS_TEST:
        print(f"\n=== [Full Model] Metrics for guide_w={w} ===")
        
        # Per-sample Means
        print("RMSE (mean):", mean_or_nan(per_sample_metrics[w]['rmse']))
        print("Trad AP (mean):", mean_or_nan(per_sample_metrics[w]['trad_ap']))
        print("Brier (mean):", mean_or_nan(per_sample_metrics[w]['brier']))
        
        # Global Aggregation
        gm = None
        try:
            preds_all = np.concatenate([p.flatten() for p in all_preds[w]])
            gts_all = np.concatenate([g.flatten() for g in all_gts[w]])
            mask = np.isfinite(preds_all) & np.isfinite(gts_all)
            preds_all = preds_all[mask]; gts_all = gts_all[mask]
            
            if preds_all.size > 0:
                rmse_global = float(np.sqrt(np.mean((preds_all - gts_all)**2)))
                brier_global = float(np.mean((preds_all - gts_all)**2))
                
                gtbin_global = (gts_all > BIN_THRESH).astype(int)
                trad_ap_global = float(average_precision_score(gtbin_global, preds_all)) if gtbin_global.sum() > 0 else float('nan')
                
                # Global Coverage
                covs = {p: coverage_topk(preds_all.reshape(-1,1)[:,0], gts_all.reshape(-1,1)[:,0], p) for p in TOP_PERCENTS}
                
                gm = {
                    'rmse_global': rmse_global,
                    'brier_global': brier_global,
                    'trad_ap_global': trad_ap_global,
                    'coverage': covs
                }
        except Exception as e:
            print("Error computing global metrics:", e)

        # 构建 CSV 行 (字段对齐)
        row = {
            'guide_w': w,
            'per_rmse_mean': mean_or_nan(per_sample_metrics[w]['rmse']),
            'per_brier_mean': mean_or_nan(per_sample_metrics[w]['brier']),
            'per_trad_ap_mean': mean_or_nan(per_sample_metrics[w]['trad_ap']),
            'per_relaxed_ap_r1_mean': mean_or_nan(per_sample_metrics[w]['relaxed_ap_r1']),
            'per_weighted_ap_mean': mean_or_nan(per_sample_metrics[w]['weighted_ap_per_sample']),
            'per_cov1_mean': mean_or_nan(per_sample_metrics[w]['coverage_1%']),
            'per_cov5_mean': mean_or_nan(per_sample_metrics[w]['coverage_5%']),
            'per_cov10_mean': mean_or_nan(per_sample_metrics[w]['coverage_10%']),
        }
        if gm is not None:
            row.update({
                'global_rmse': gm['rmse_global'],
                'global_brier': gm['brier_global'],
                'global_trad_ap': gm['trad_ap_global'],
                'global_cov1': gm['coverage'].get(0.01, float('nan')),
                'global_cov5': gm['coverage'].get(0.05, float('nan')),
                'global_cov10': gm['coverage'].get(0.10, float('nan')),
            })
        summary_rows.append(row)

        # --- 绘图 1: Quantile Sensitivity ---
        per_means = []
        for q in QUANTILES:
            arr = np.array([x for x in quantile_results_per_sample_mean[w][q] if not (x is None or (isinstance(x, float) and np.isnan(x)))])
            per_means.append(float(arr.mean()) if arr.size > 0 else float('nan'))
        
        plt.figure(figsize=(8,5))
        plt.plot(QUANTILES, per_means, marker='o', label='Full Model')
        plt.xlabel('Quantile')
        plt.ylabel('Average Precision')
        plt.title(f'Quantile Sensitivity (Full Model) w={w}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(SAVE_DIR, f'quantile_sensitivity_w{w}.png'))
        plt.close()

        # --- 绘图 2: Coverage Curve ---
        if gm is not None:
            coverages = []
            for tp in COVERAGE_TOP_PCT_RANGE:
                N_all = len(preds_all)
                k = max(1, int(N_all * tp))
                idx_top = np.argsort(preds_all)[::-1][:k]
                if gts_all.sum() == 0:
                    coverages.append(float('nan'))
                else:
                    coverages.append(float(gts_all[idx_top].sum() / gts_all.sum()))
            
            plt.figure(figsize=(8,5))
            plt.plot(COVERAGE_TOP_PCT_RANGE*100, coverages, label='Coverage Curve')
            plt.xlabel('Top Area (%)')
            plt.ylabel('Coverage')
            plt.title(f'Coverage Curve (Full Model) w={w}')
            plt.grid(True)
            plt.savefig(os.path.join(SAVE_DIR, f'coverage_curve_w{w}.png'))
            plt.close()

    # 保存 Summary CSV
    csv_path = os.path.join(SAVE_DIR, "evaluation_summary.csv")
    keys = sorted(summary_rows[0].keys()) if summary_rows else []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    
    print(f"Done. Results and plots saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()