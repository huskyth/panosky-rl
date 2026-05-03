from entries.phalanx.phalanx import Phalanx
from factory.config_factory import ConfigFactory, ConfigEnum


class PhalanxFactory:
    phalanx_instance = None

    @staticmethod
    def create(*keys, **kward):
        if PhalanxFactory.phalanx_instance is not None:
            return PhalanxFactory.phalanx_instance
        PhalanxFactory.phalanx_instance = Phalanx(ConfigFactory.create(*keys, config_type=ConfigEnum.phalanx, **kward))
        return PhalanxFactory.phalanx_instance

    @staticmethod
    def reset():
        PhalanxFactory.phalanx_instance = None


if __name__ == '__main__':
    p1 = PhalanxFactory.create()
    p2 = PhalanxFactory.create()
    print(p1, p2, p1 is p2, id(p1) == id(p2))
