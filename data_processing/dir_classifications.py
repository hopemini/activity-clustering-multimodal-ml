import os
from tqdm import tqdm
import argparse

par = argparse.ArgumentParser()
par.add_argument("-t", "--type", default='jpg',
                 type=str, help="image type(jpg/png)")
args = par.parse_args()

for i in tqdm(range(75000)):
    if os.path.isfile(str(i)+'.' + args.type):
        os.system('mkdir '+str(i))
        os.system('mv '+str(i)+'.' + args.type + ' '+str(i)+'/')
