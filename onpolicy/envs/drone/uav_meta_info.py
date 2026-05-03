import numpy as np


class TrainUAV:
    def __init__(self, px, py, pz, vx, vy, vz, is_attacked_state, status):
        self.position = [px, py, pz]
        self.velocity = [vx, vy, vz]
        self.is_attacked_state = is_attacked_state
        self.status = status

    def get_position_and_velocity(self):
        return self.position + self.velocity

    def get_normalize_velocity(self, game_velocity):
        return (np.array(self.velocity) / game_velocity).tolist()

    def get_normalize_position(self, normal_):
        return (np.array(self.position) / normal_).tolist()

    def set_position(self, x, y, z):
        self.position = [x, y, z]

    def set_velocity(self, x, y, z):
        self.velocity = [x, y, z]

    def to_dict(self):
        return {
            "position": self.position,
            "velocity": self.velocity,
            "status": self.status.value,
            'is_attacked_state': str(self.is_attacked_state)
        }
