import pickle
import json
import numpy as np
from dataclasses import asdict
from dataclasses_json import dataclass_json
#from src.dto.dto_generator import TrainingFormatData

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Render an image from training format data.')
    parser.add_argument('--load_filename', type=str,default='gen_data/load_eng_tmp/metadata/0_0.pkl',help='filename for loading')
    parser.add_argument('--save_filename', type=str,default='example/tfd.json',help='filename for saving')
    args,_ = parser.parse_known_args()
    tfd = pickle.load(open(args.load_filename, 'rb'))
    _tfd_dict = asdict(tfd)
    #tfd = TrainingFormatData(**tfd_dict) dict to dto
    #tfd_dict = {k:v.tolist() for k,v in tfd_dict.items()}
    tfd_dict = {}
    for k,v in _tfd_dict.items():
        if type(v)==np.ndarray:
            tfd_dict[k]=v.tolist()
        else:
            tfd_dict[k]=v
    with open(args.save_filename, 'w') as fp:
        json.dump(tfd_dict, fp)
