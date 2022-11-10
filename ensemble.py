import numpy as np
import sys
from glob import glob
from ogb.lsc import PCQM4Mv2Evaluator

from OGB_ViSNet.postprocess import postprocess, rewrite


dir_path, csv_file_path, split_file_path = sys.argv[1:]

if __name__ == '__main__':
    
    pt_filepaths = glob("results/test-challenge-pt-visnet-*")
    pt_y_preds = []
    for filepath in pt_filepaths:
        pt_y_preds.append(np.load(f"{filepath}/y_pred_pcqm4m-v2_test-challenge.npz")["y_pred"])
    pt_y_preds = np.stack(pt_y_preds, axis=0)
    
    tm_filepaths = glob("results/test-challenge-tm-visnet-*")
    tm_y_preds = []
    for filepath in tm_filepaths:
        tm_y_preds.append(np.load(f"{filepath}/y_pred_pcqm4m-v2_test-challenge.npz")["y_pred"])
    tm_y_preds = np.stack(tm_y_preds, axis=0)
    
    y_pred = np.concatenate([pt_y_preds, tm_y_preds], axis=0)
    y_preds_sorted = np.sort(y_pred, axis=0)
    y_pred = np.mean(y_preds_sorted[6:-6], axis=0)

    # Post process
    gaps = postprocess(csv_file=csv_file_path, split_file=split_file_path)
    y_pred = rewrite(y_pred, gaps, csv_file=csv_file_path, split_file=split_file_path)
    
    evaluator = PCQM4Mv2Evaluator()
    input_dict = {'y_pred': y_pred}
    evaluator.save_test_submission(input_dict=input_dict, dir_path=dir_path, mode='test-challenge')
    