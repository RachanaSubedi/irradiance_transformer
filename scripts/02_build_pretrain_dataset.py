import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from irradiance.config import CFG
from irradiance.data import build_pretrain_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true')
args = parser.parse_args()

build_pretrain_dataset(CFG, force=args.force)
