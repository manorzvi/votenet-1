import numpy as np

from typing import List, Optional


class ShapenetTransforms:

    coords = {"x": 0, "y": 1, "z": 2}
    rotation_matrices = {
        "xy": lambda x: ShapenetTransforms.rotz(x),
        "xz": lambda x: ShapenetTransforms.roty(x),
        "yz": lambda x: ShapenetTransforms.rotx(x),
    }

    @staticmethod
    def to_center(pc: np.ndarray):
        center = np.mean(pc, axis=0, keepdims=True)
        pc -= center
        return pc

    @staticmethod
    def on_floor(pc: np.ndarray, floor_normal: str = "z") -> np.ndarray:
        floor_normal = ShapenetTransforms.coords[floor_normal]
        height_from_floor = np.min(pc[:, floor_normal])
        floor_translation = np.zeros((1, 3))
        floor_translation[0, floor_normal] = height_from_floor
        pc -= floor_translation
        return pc

    @staticmethod
    def to_rotate(pc: np.ndarray, alpha: float, orientation: str = "xy") -> np.ndarray:
        rotation_matrix = ShapenetTransforms.rotation_matrices[orientation](alpha)
        pc = pc @ rotation_matrix
        return pc

    @staticmethod
    def to_translate(pc: np.ndarray, t_vec: np.ndarray, step_size: Optional[float] = 1.0) -> np.ndarray:
        t_vec *= step_size
        pc += t_vec
        return pc

    @staticmethod
    def to_standard(pc: np.ndarray) -> np.ndarray:
        pc = ShapenetTransforms.to_center(pc)
        pc = ShapenetTransforms.to_rotate(pc, alpha=-np.pi/2, orientation='yz')
        pc = ShapenetTransforms.on_floor(pc)
        return pc

    @staticmethod
    def rotx(t: float):
        """Rotation about the x-axis."""
        c = np.cos(t).item()
        s = np.sin(t).item()
        rot_mat = np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return rot_mat

    @staticmethod
    def roty(t: float):
        """Rotation about the y-axis."""
        c = np.cos(t).item()
        s = np.sin(t).item()
        return np.ndarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    @staticmethod
    def rotz(t: float):
        """Rotation about the z-axis."""
        c = np.cos(t).item()
        s = np.sin(t).item()
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def rand_alpha():
        return 2 * np.pi * np.random.rand(1)

    @staticmethod
    def rand_unit2_vector(coord_to_discard: List[str] = ['z']):
        v = 2 * np.random.rand(3, ) - 1
        for coord in coord_to_discard:
            zero_direction = ShapenetTransforms.coords[coord]
            v[zero_direction] = 0
        return v / np.linalg.norm(v)
