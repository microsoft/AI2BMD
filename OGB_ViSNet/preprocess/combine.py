import os
import glob
import pickle
import sys
from tqdm import tqdm
from ogb.lsc import PCQM4Mv2Dataset

DATA_ROOT, TC = sys.argv[1:]

pickle_files = glob.glob(os.path.join(DATA_ROOT, "*.pickle"))
pickle_files.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))

whole_data = []

for file in tqdm(pickle_files, desc='Combining'):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    whole_data.extend(data)   

save_name = "whole.pkl" if TC == "False" else "whole_tc.pkl"
with open(os.path.join(DATA_ROOT, save_name), 'wb') as f:
    pickle.dump(whole_data, f)