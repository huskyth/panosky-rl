from pathlib import Path

from onpolicy.scripts.train.train_drone import parse_args
import sys

from onpolicy.envs.drone.mul_uav_env import MultiUavEnv

temp = Path(__file__).parent.parent
sys.path.append(str(temp))
from onpolicy.config import get_config

"""Train script for MPEs."""
import configparser

# 读取配置文件中的参数并保存
config_path = temp / 'config/th_demo.ini'
cf = configparser.ConfigParser()
cf.read(str(config_path), encoding="utf-8")

parser = get_config()
all_args = parse_args(sys.argv[1:], parser)
env = MultiUavEnv(cf=cf, is_debug=True)
import numpy as np

env.reset()
while not env.is_terminal[0]:
    env.step(np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]))
