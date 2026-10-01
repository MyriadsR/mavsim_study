"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        1/13/2021 - TWM
        7/13/2023 - RWB
        1/16/2024 - RWB
"""
import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import euler_to_rotation
from tools.drawing import rotate_points, translate_points, points_to_mesh


class DrawMav:
    def __init__(self, state, window, scale=10):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.north  # north position
            state.east  # east position
            state.altitude   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        self.unit_length = scale
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        # convert North-East Down to East-North-Up for rendering
        self.R_ned = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        # each part of the uav is drawn as its own triangular mesh
        self.parts = [
            self.get_body_points(),
            self.get_wing_points(),
            self.get_tail_points(),
            self.get_prop_points(),
        ]
        self.mav_objects = []
        for points, index, meshColors in self.parts:
            object = self.add_object(points, index, meshColors, R_bi, mav_position)
            window.addItem(object)  # add part to plot
            self.mav_objects.append(object)

    def update(self, state):
        mav_position = np.array([[state.north], [state.east], [-state.altitude]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R_bi = euler_to_rotation(state.phi, state.theta, state.psi)
        for object, (points, index, meshColors) in zip(self.mav_objects, self.parts):
            self.update_object(object, points, index, meshColors, R_bi, mav_position)

    def add_object(self, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object = gl.GLMeshItem(
            vertexes=mesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=colors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        return object

    def update_object(self, object, points, index, colors, R, position):
        rotated_points = rotate_points(points, R)
        translated_points = translate_points(rotated_points, position)
        translated_points = self.R_ned @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = points_to_mesh(translated_points, index)
        object.setMeshData(vertexes=mesh, vertexColors=colors)
        return object

    def _box(self, x, y, z):
        """
        vertices and triangular faces of an axis-aligned box, defined in the
        body frame (x out the nose, y out the right wing, z out the belly),
        with x=(x_min, x_max), y=(y_min, y_max), z=(z_min, z_max)
        """
        x0, x1 = x
        y0, y1 = y
        z0, z1 = z
        points = self.unit_length * np.array([
            [x1, y0, z0],  # point 0
            [x1, y1, z0],  # point 1
            [x0, y1, z0],  # point 2
            [x0, y0, z0],  # point 3
            [x1, y0, z1],  # point 4
            [x1, y1, z1],  # point 5
            [x0, y1, z1],  # point 6
            [x0, y0, z1],  # point 7
        ]).T
        index = np.array([
            [0, 1, 2], [0, 2, 3],  # top
            [4, 6, 5], [4, 7, 6],  # bottom
            [0, 4, 5], [0, 5, 1],  # nose face
            [3, 2, 6], [3, 6, 7],  # tail face
            [1, 5, 6], [1, 6, 2],  # right side
            [0, 3, 7], [0, 7, 4],  # left side
        ])
        return points, index

    def _colors(self, color, n_faces=12):
        meshColors = np.empty((n_faces, 3, 4), dtype=np.float32)
        meshColors[:] = color
        return meshColors

    def get_body_points(self):
        """ fuselage: elongated box along the body x axis """
        points, index = self._box(x=(-1.6, 1.6), y=(-0.18, 0.18), z=(-0.15, 0.15))
        red = np.array([1., 0., 0., 1])
        return points, index, self._colors(red)

    def get_wing_points(self):
        """ main wing: thin, wide box spanning the body y axis """
        points, index = self._box(x=(-0.5, 0.5), y=(-3.2, 3.2), z=(-0.06, 0.06))
        blue = np.array([0., 0., 1., 1])
        return points, index, self._colors(blue)

    def get_tail_points(self):
        """ empennage: horizontal stabilizer plus vertical stabilizer """
        points_h, index_h = self._box(x=(-2.2, -1.4), y=(-1.1, 1.1), z=(-0.05, 0.05))
        points_v, index_v = self._box(x=(-2.2, -1.4), y=(-0.05, 0.05), z=(-1.0, 0.0))
        points = np.concatenate((points_h, points_v), axis=1)
        # renumber the vertical tail faces to point at the appended vertices
        index_v = index_v + points_h.shape[1]
        index = np.concatenate((index_h, index_v), axis=0)
        green = np.array([0., 1., 0., 1])
        return points, index, self._colors(green, n_faces=24)

    def get_prop_points(self):
        """ propeller: two thin blades crossed at the nose """
        points_a, index_a = self._box(x=(1.6, 1.7), y=(-1.0, 1.0), z=(-0.04, 0.04))
        points_b, index_b = self._box(x=(1.6, 1.7), y=(-0.04, 0.04), z=(-1.0, 1.0))
        points = np.concatenate((points_a, points_b), axis=1)
        index_b = index_b + points_a.shape[1]
        index = np.concatenate((index_a, index_b), axis=0)
        yellow = np.array([1., 1., 0., 1])
        return points, index, self._colors(yellow, n_faces=24)
