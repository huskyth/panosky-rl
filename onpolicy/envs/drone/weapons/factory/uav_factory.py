from entries.uav.uav import UAV
from factory.config_factory import ConfigFactory, ConfigEnum


class UAVFactory:
    @staticmethod
    def create():
        return UAV(ConfigFactory.create(config_type=ConfigEnum.uav))
