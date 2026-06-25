import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))
from irradiance.config import CFG
from irradiance.data import build_master

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true')
args = parser.parse_args()

master = build_master(CFG, force=args.force)
print(master.shape)
print(CFG.paths['master'])
