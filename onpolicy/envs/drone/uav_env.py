from onpolicy.envs.drone.mul_uav_env import MultiUavEnv, logger
import configparser
from pathlib import Path


def uav_env(args, mode):
    config_path = Path(__file__).parent / f'config/{args.ini_name}.ini'
    cf = configparser.ConfigParser()
    cf.read(str(config_path), encoding="utf-8")
    return MultiUavEnv(mode, cf)
