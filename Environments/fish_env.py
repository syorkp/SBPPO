import random
import matplotlib.pyplot as plt
import pymunk
from skimage.transform import resize, rescale
import numpy as np
from skimage.draw import line
from Tools.drawing_board import DrawingBoard
from Environments.Action_Space.draw_angle_dist import draw_angle_dist


class BaseEnvironment:
    """A base class to represent environments, for extension to ProjectionEnvironment, VVR and Naturalistic
    environment classes."""

    def __init__(self, env_variables, draw_screen):
        self.env_variables = env_variables
        self.board = DrawingBoard(self.env_variables['width'], self.env_variables['height'])
        self.draw_screen = draw_screen
        self.show_all = False
        self.num_steps = 0
        self.fish = None

        if self.draw_screen:
            self.board_fig, self.ax_board = plt.subplots()
            self.board_image = plt.imshow(np.zeros((self.env_variables['height'], self.env_variables['width'], 3)))
            plt.ion()
            plt.show()

        self.dark_col = int(self.env_variables['width'] * self.env_variables['dark_light_ratio'])
        if self.dark_col == 0:  # Fixes bug with left wall always being invisible.
            self.dark_col = -1

        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0.0, 0.0)
        self.space.damping = self.env_variables['drag']

        self.prey_bodies = []
        self.prey_shapes = []

        self.prey_cloud_wall_shapes = []
        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['width'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120),
                 np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['height'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120)]
                for cloud in range(self.env_variables["prey_cloud_num"])]
        self.predator_bodies = []
        self.predator_shapes = []

        self.predator_body = None
        self.predator_shape = None
        self.predator_target = None

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.last_action = None

        self.vegetation_bodies = []
        self.vegetation_shapes = []

        self.background = None

        self.prey_consumed_this_step = False

        self.predators_avoided = 0
        self.prey_caught = 0
        self.sand_grains_bumped = 0
        self.steps_near_vegetation = 0

        self.stimuli_information = {}

    def reset(self):
        self.num_steps = 0
        self.fish.hungry = 0
        self.fish.stress = 1
        self.prey_caught = 0
        self.predators_avoided = 0
        self.sand_grains_bumped = 0
        self.steps_near_vegetation = 0

        for i, shp in enumerate(self.prey_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.sand_grain_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.vegetation_shapes):
            self.space.remove(shp, shp.body)

        for i, shp in enumerate(self.prey_cloud_wall_shapes):
            self.space.remove(shp)

        self.prey_cloud_wall_shapes = []

        if self.predator_shape is not None:
            self.remove_realistic_predator()

        self.prey_shapes = []
        self.prey_bodies = []

        self.predator_shapes = []
        self.predator_bodies = []

        self.sand_grain_shapes = []
        self.sand_grain_bodies = []

        self.vegetation_bodies = []
        self.vegetation_shapes = []

    def output_frame(self, activations, internal_state, scale=0.25):
        arena = self.board.db * 255.0
        arena[0, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[self.env_variables['height'] - 1, :, 0] = np.ones(self.env_variables['width']) * 255
        arena[:, 0, 0] = np.ones(self.env_variables['height']) * 255
        arena[:, self.env_variables['width'] - 1, 0] = np.ones(self.env_variables['height']) * 255

        eyes = self.fish.get_visual_inputs()

        frame = np.vstack((arena, np.zeros((50, self.env_variables['width'], 3)), eyes))

        this_ac = np.zeros((20, self.env_variables['width'], 3))
        this_ac[:, :, 0] = resize(internal_state, (20, self.env_variables['width']), anti_aliasing=False, order=0) * 255
        this_ac[:, :, 1] = resize(internal_state, (20, self.env_variables['width']), anti_aliasing=False, order=0) * 255
        this_ac[:, :, 2] = resize(internal_state, (20, self.env_variables['width']), anti_aliasing=False, order=0) * 255

        frame = np.vstack((frame, np.zeros((20, self.env_variables['width'], 3)), this_ac))

        if activations is not None:

            adr = [-1, 1]
            for ac in range(len(activations)):
                this_ac = np.zeros((20, self.env_variables['width'], 3))
                pos = (activations[ac] - adr[0]) / (adr[1] - adr[0])

                pos[pos < 0] = 0
                pos[pos > 1] = 1

                this_ac[:, :, 0] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 1] = resize(pos, (20, self.env_variables['width'])) * 255
                this_ac[:, :, 2] = resize(pos, (20, self.env_variables['width'])) * 255

                frame = np.vstack((frame, np.zeros((20, self.env_variables['width'], 3)), this_ac))

        frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)
        return frame

    def draw_shapes(self):
        self.board.fish_shape(self.fish.body.position, self.env_variables['fish_mouth_size'],
                              self.env_variables['fish_head_size'], self.env_variables['fish_tail_length'],
                              self.fish.mouth.color, self.fish.head.color, self.fish.body.angle)

        if len(self.prey_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.prey_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.prey_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, self.env_variables['prey_size'])

            try:
                self.board.db[rrs, ccs] = self.prey_shapes[0].color
            except IndexError:
                print(f"Index Error for: PX: {max(rrs.flatten())}, PY: {max(ccs.flatten())}")
                if max(rrs.flatten()) > self.env_variables['height']:
                    lost_index = np.argmax(py)
                elif max(ccs.flatten()) > self.env_variables['width']:
                    lost_index = np.argmax(px)
                else:
                    lost_index = 0
                    print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
                self.prey_bodies.pop(lost_index)
                self.prey_shapes.pop(lost_index)
                self.draw_shapes()

        if len(self.sand_grain_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.sand_grain_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.sand_grain_bodies])).astype(int)
            rrs, ccs = self.board.multi_circles(px, py, self.env_variables['sand_grain_size'])

            try:
                self.board.db[rrs, ccs] = self.sand_grain_shapes[0].color
            except IndexError:
                print(f"Index Error for: RRS: {max(rrs.flatten())}, CCS: {max(ccs.flatten())}")
                if max(rrs.flatten()) > self.env_variables['width']:
                    lost_index = np.argmax(px)
                elif max(ccs.flatten()) > self.env_variables['height']:
                    lost_index = np.argmax(py)
                else:
                    lost_index = 0
                    print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
                self.sand_grain_bodies.pop(lost_index)
                self.sand_grain_shapes.pop(lost_index)
                self.draw_shapes()

        for i, pr in enumerate(self.predator_bodies):
            self.board.circle(pr.position, self.env_variables['predator_size'], self.predator_shapes[i].color)

        for i, pr in enumerate(self.vegetation_bodies):
            self.board.vegetation(pr.position, self.env_variables['vegetation_size'], self.vegetation_shapes[i].color)

        if self.predator_body is not None:
            self.board.circle(self.predator_body.position, self.env_variables['predator_size'],
                              self.predator_shape.color)

        if self.background:
            if self.background == "Green":
                colour = (0, 1, 0)
            elif self.background == "Red":
                colour = (1, 0, 0)
            else:
                print("Invalid Background Colour")
                return
            self.board.create_screen(self.fish.body.position, self.env_variables["max_vis_dist"], colour)

    def build_prey_cloud_walls(self):
        for i in self.prey_cloud_locations:
            wall_edges = [
            pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] - 150), (i[0] - 150, i[1] + 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] + 150), (i[0] + 150, i[1] + 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] + 150, i[1] + 150), (i[0] + 150, i[1] - 150), 1),
                pymunk.Segment(
                    self.space.static_body,
                    (i[0] - 150, i[1] - 150), (i[0] + 150, i[1] - 150), 1)
            ]
            for s in wall_edges:
                s.friction = 1.
                s.group = 1
                s.collision_type = 7
                s.color = (0, 0, 0)
                self.space.add(s)
                self.prey_cloud_wall_shapes.append(s)

    def create_walls(self):
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, self.env_variables['height']), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, self.env_variables['height']), (self.env_variables['width'], self.env_variables['height']), 1),
            pymunk.Segment(
                self.space.static_body,
                (self.env_variables['width'] - 1, self.env_variables['height']), (self.env_variables['width'] - 1, 1),
                1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (self.env_variables['width'], 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = (1, 0, 0)
            self.space.add(s)

        # self.space.add(static)

    @staticmethod
    def no_collision(arbiter, space, data):
        return False

    def touch_edge(self, arbiter, space, data):
        new_position_x = self.fish.body.position[0]
        new_position_y = self.fish.body.position[1]
        if new_position_x < 30:  # Wall d
            new_position_x += self.env_variables["fish_head_size"] + self.env_variables["fish_tail_length"]
        elif new_position_x > self.env_variables['width'] - 30:  # wall b
            new_position_x -= self.env_variables["fish_head_size"] + self.env_variables["fish_tail_length"]
        if new_position_y < 30:  # wall a
            new_position_y += self.env_variables["fish_head_size"] + self.env_variables["fish_tail_length"]
        elif new_position_y > self.env_variables['height'] - 30:  # wall c
            new_position_y -= self.env_variables["fish_head_size"] + self.env_variables["fish_tail_length"]

        new_position = pymunk.Vec2d(new_position_x, new_position_y)
        self.fish.body.position = new_position
        self.fish.body.velocity = (0, 0)

        if self.fish.body.angle < np.pi:
            self.fish.body.angle += np.pi
        else:
            self.fish.body.angle -= np.pi
        self.fish.touched_edge = True
        return True

    def create_prey(self):
        self.prey_bodies.append(pymunk.Body(self.env_variables['prey_mass'], self.env_variables['prey_inertia']))
        self.prey_shapes.append(pymunk.Circle(self.prey_bodies[-1], self.env_variables['prey_size']))
        self.prey_shapes[-1].elasticity = 1.0
        if not self.env_variables["differential_prey"]:
            self.prey_bodies[-1].position = (
                np.random.randint(self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                  self.env_variables['width'] - (
                                          self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'])),
                np.random.randint(self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                  self.env_variables['height'] - (
                                          self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'])))
        else:
            cloud = random.choice(self.prey_cloud_locations)
            self.prey_bodies[-1].position = (
                np.random.randint(low=cloud[0] - 120, high=cloud[0] + 120),
                np.random.randint(low=cloud[1] - 120, high=cloud[1] + 120)
            )
        self.prey_shapes[-1].color = (0, 0, 1)
        self.prey_shapes[-1].collision_type = 2
        self.prey_shapes[-1].filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.prey_bodies[-1], self.prey_shapes[-1])

    def check_proximity(self, feature_position, sensing_distance):
        sensing_area = [[feature_position[0] - sensing_distance,
                         feature_position[0] + sensing_distance],
                        [feature_position[1] - sensing_distance,
                         feature_position[1] + sensing_distance]]
        is_in_area = sensing_area[0][0] <= self.fish.body.position[0] <= sensing_area[0][1] and \
                     sensing_area[1][0] <= self.fish.body.position[1] <= sensing_area[1][1]
        if is_in_area:
            return True
        else:
            return False

    def move_prey(self):
        # Not, currently, a prey isn't guaranteed to try to escape if a loud predator is near, only if it was going to
        # move anyway. Should reconsider this in the future.
        to_move = np.where(np.random.rand(len(self.prey_bodies)) < self.env_variables['prey_impulse_rate'])[0]
        for ii in range(len(to_move)):
            if self.check_proximity(self.prey_bodies[to_move[ii]].position,
                                    self.env_variables['prey_sensing_distance']) and self.env_variables["prey_jump"]:
                self.prey_bodies[ii].angle = self.fish.body.angle + np.random.uniform(-1, 1)
                self.prey_bodies[to_move[ii]].apply_impulse_at_local_point((self.get_last_action_magnitude(), 0))
            else:
                adjustment = np.random.uniform(-self.env_variables['prey_max_turning_angle'],
                                               self.env_variables['prey_max_turning_angle'])
                self.prey_bodies[to_move[ii]].angle = self.prey_bodies[to_move[ii]].angle + adjustment
                self.prey_bodies[to_move[ii]].apply_impulse_at_local_point((self.env_variables['prey_impulse'], 0))

    def touch_prey(self, arbiter, space, data):
        if self.fish.making_capture:
            for i, shp in enumerate(self.prey_shapes):
                if shp == arbiter.shapes[0]:
                    space.remove(shp, shp.body)
                    self.prey_shapes.remove(shp)
                    self.prey_bodies.remove(shp.body)
            self.prey_caught += 1
            self.fish.prey_consumed = True
            self.prey_consumed_this_step = True

            return False
        else:
            return True

    def create_predator(self):
        self.predator_bodies.append(
            pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia']))
        self.predator_shapes.append(pymunk.Circle(self.predator_bodies[-1], self.env_variables['predator_size']))
        self.predator_shapes[-1].elasticity = 1.0
        self.predator_bodies[-1].position = (
            np.random.randint(self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['width'] - (
                                      self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'])),
            np.random.randint(self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['height'] - (
                                      self.env_variables['predator_size'] + self.env_variables['fish_mouth_size'])))
        self.predator_shapes[-1].color = (0, 0, 1)
        self.predator_shapes[-1].collision_type = 5

        self.space.add(self.predator_bodies[-1], self.predator_shapes[-1])

    def move_predator(self):
        for pr in self.predator_bodies:
            dist_to_fish = np.sqrt(
                (pr.position[0] - self.fish.body.position[0]) ** 2 + (pr.position[1] - self.fish.body.position[1]) ** 2)

            if dist_to_fish < self.env_variables['predator_sensing_dist']:
                pr.angle = np.pi / 2 - np.arctan2(self.fish.body.position[0] - pr.position[0],
                                                  self.fish.body.position[1] - pr.position[1])
                pr.apply_impulse_at_local_point((self.env_variables['predator_chase_impulse'], 0))

            elif np.random.rand(1) < self.env_variables['predator_impulse_rate']:
                pr.angle = np.random.rand(1) * 2 * np.pi
                pr.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def touch_predator(self, arbiter, space, data):
        if self.num_steps > self.env_variables['immunity_steps']:
            self.fish.touched_predator = True
            return False
        else:
            return True

    def check_fish_proximity_to_walls(self):
        fish_position = self.fish.body.position

        # Check proximity to left wall
        if 0 < fish_position[0] < self.env_variables["distance_from_fish"]:
            left = True
        else:
            left = False

        # Check proximity to right wall
        if self.env_variables["width"] - self.env_variables["distance_from_fish"] < fish_position[0] < \
                self.env_variables["width"]:
            right = True
        else:
            right = False

        # Check proximity to bottom wall
        if self.env_variables["height"] - self.env_variables["distance_from_fish"] < fish_position[1] < \
                self.env_variables["height"]:
            bottom = True
        else:
            bottom = False

        # Check proximity to top wall
        if 0 < fish_position[0] < self.env_variables["distance_from_fish"]:
            top = True
        else:
            top = False

        return left, bottom, right, top

    def select_predator_angle_of_attack(self):
        left, bottom, right, top = self.check_fish_proximity_to_walls()
        if left and top:
            angle_from_fish = random.randint(90, 180)
        elif left and bottom:
            angle_from_fish = random.randint(0, 90)
        elif right and top:
            angle_from_fish = random.randint(180, 270)
        elif right and bottom:
            angle_from_fish = random.randint(270, 360)
        elif left:
            angle_from_fish = random.randint(0, 180)
        elif top:
            angle_from_fish = random.randint(90, 270)
        elif bottom:
            angles = [random.randint(270, 360), random.randint(0, 90)]
            angle_from_fish = random.choice(angles)
        elif right:
            angle_from_fish = random.randint(180, 360)
        else:
            angle_from_fish = random.randint(0, 360)

        angle_from_fish = np.radians(angle_from_fish / np.pi)
        return angle_from_fish

    def create_realistic_predator(self):
        self.predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])
        self.predator_shape = pymunk.Circle(self.predator_body, self.env_variables['predator_size'])
        self.predator_shape.elasticity = 1.0

        fish_position = self.fish.body.position

        angle_from_fish = self.select_predator_angle_of_attack()
        dy = self.env_variables["distance_from_fish"] * np.cos(angle_from_fish)
        dx = self.env_variables["distance_from_fish"] * np.sin(angle_from_fish)

        x_position = fish_position[0] + dx
        y_position = fish_position[1] + dy

        self.predator_body.position = (x_position, y_position)
        self.predator_target = fish_position  # Update so appears where fish will be in a few steps.

        self.predator_shape.color = (0, 0, 1)
        self.predator_shape.collision_type = 5
        self.predator_shape.filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # Category 2 objects cant collide with predator

        self.space.add(self.predator_body, self.predator_shape)

    def check_predator_inside_walls(self):
        x_position, y_position = self.predator_body.position[0], self.predator_body.position[1]
        if x_position < 0:
            return True
        elif x_position > self.env_variables["width"]:
            return True
        if y_position < 0:
            return True
        elif y_position > self.env_variables["height"]:
            return True

    def check_predator_at_target(self):
        if (round(self.predator_body.position[0]), round(self.predator_body.position[1])) == (
                round(self.predator_target[0]), round(self.predator_target[1])):
            return True
        else:
            return False

    def move_realistic_predator(self):
        if self.check_predator_at_target():
            self.remove_realistic_predator()
            self.predators_avoided += 1
            return
        if self.check_predator_inside_walls():
            self.remove_realistic_predator()
            return

        self.predator_body.angle = np.pi / 2 - np.arctan2(
            self.predator_target[0] - self.predator_body.position[0],
            self.predator_target[1] - self.predator_body.position[1])
        self.predator_body.apply_impulse_at_local_point((self.env_variables['predator_impulse'], 0))

    def remove_realistic_predator(self, arbiter=None, space=None, data=None):
        if self.predator_body is not None:
            self.space.remove(self.predator_shape, self.predator_shape.body)
            self.predator_shape = None
            self.predator_body = None
        else:
            pass

    def create_sand_grain(self):
        self.sand_grain_bodies.append(
            pymunk.Body(self.env_variables['sand_grain_mass'], self.env_variables['sand_grain_inertia']))
        self.sand_grain_shapes.append(pymunk.Circle(self.sand_grain_bodies[-1], self.env_variables['sand_grain_size']))
        self.sand_grain_shapes[-1].elasticity = 1.0
        self.sand_grain_bodies[-1].position = (
            np.random.randint(self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['width'] - (
                                      self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'])),
            np.random.randint(self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['height'] - (
                                      self.env_variables['sand_grain_size'] + self.env_variables['fish_mouth_size'])))
        self.sand_grain_shapes[-1].color = (0, 0, 1)
        self.sand_grain_shapes[-1].collision_type = 4
        self.sand_grain_shapes[-1].filter = pymunk.ShapeFilter(
            mask=pymunk.ShapeFilter.ALL_MASKS ^ 2)  # prevents collisions with predator

        self.space.add(self.sand_grain_bodies[-1], self.sand_grain_shapes[-1])

    def touch_grain(self, arbiter, space, data):
        # TODO: Considering only doing this if the last swim was a capture swim.
        if self.last_action == 3:
            self.sand_grains_bumped += 1

    def get_last_action_magnitude(self):
        return self.fish.prev_action_impulse / 200  # Scaled down both for mass effects and to make it possible for the prey to be caught. TODO: Consider making this a parameter.

    def displace_sand_grains(self):
        for i, body in enumerate(self.sand_grain_bodies):
            if self.check_proximity(self.sand_grain_bodies[i].position,
                                    self.env_variables['sand_grain_displacement_distance']):
                self.sand_grain_bodies[i].angle = self.fish.body.angle + np.random.uniform(-1, 1)
                # if self.sand_grain_bodies[i].angle < (3 * np.pi) / 2:
                #     self.sand_grain_bodies[i].angle += np.pi / 2
                # else:
                #     self.sand_grain_bodies[i].angle -= np.pi / 2
                self.sand_grain_bodies[i].apply_impulse_at_local_point(
                    (self.get_last_action_magnitude(), 0))

    def create_vegetation(self):
        size = self.env_variables['vegetation_size']
        vertices = [(0, 0), (0, size), (size / 2, size - size / 3), (size, size), (size, 0), (size / 2, size / 3)]
        self.vegetation_bodies.append(pymunk.Body(body_type=pymunk.Body.STATIC))
        self.vegetation_shapes.append(pymunk.Poly(self.vegetation_bodies[-1], vertices))
        self.vegetation_bodies[-1].position = (
            np.random.randint(self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['width'] - (
                                      self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'])),
            np.random.randint(self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'],
                              self.env_variables['height'] - (
                                      self.env_variables['vegetation_size'] + self.env_variables['fish_mouth_size'])))
        self.vegetation_shapes[-1].color = (0, 1, 0)
        self.vegetation_shapes[-1].collision_type = 1
        self.vegetation_shapes[-1].friction = 1

        self.space.add(self.vegetation_bodies[-1], self.vegetation_shapes[-1])

    def check_fish_near_vegetation(self):
        vegetation_locations = [v.position for v in self.vegetation_bodies]
        fish_surrounding_area = [[self.fish.body.position[0] - self.env_variables['vegetation_effect_distance'],
                                  self.fish.body.position[0] + self.env_variables['vegetation_effect_distance']],
                                 [self.fish.body.position[1] - self.env_variables['vegetation_effect_distance'],
                                  self.fish.body.position[1] + self.env_variables['vegetation_effect_distance']]]
        for veg in vegetation_locations:
            is_in_area = fish_surrounding_area[0][0] <= veg[0] <= fish_surrounding_area[0][1] and \
                         fish_surrounding_area[1][0] <= veg[1] <= fish_surrounding_area[1][1]
            if is_in_area:
                self.steps_near_vegetation += 1
                return True
        return False


class ContinuousNaturalisticEnvironment(BaseEnvironment):

    def __init__(self, env_variables, realistic_bouts, draw_screen=False, fish_mass=None, collisions=True):
        super().__init__(env_variables, draw_screen)

        # Create the fish class instance and add to the space.
        if fish_mass is None:
            self.fish = ContinuousFish(self.board, env_variables, self.dark_col, realistic_bouts)
        else:
            # In the event that I am producing a calibration curve for distance moved.
            self.fish = ContinuousFish(self.board, env_variables, self.dark_col, realistic_bouts, fish_mass=fish_mass)

        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)

        # Create walls.
        self.create_walls()
        self.reset()

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        if collisions:
            self.pred_col = self.space.add_collision_handler(5, 3)
            self.pred_col.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

        self.edge_pred_col = self.space.add_collision_handler(1, 5)
        self.edge_pred_col.begin = self.remove_realistic_predator

        self.grain_fish_col = self.space.add_collision_handler(3, 4)
        self.grain_fish_col.begin = self.touch_grain

        # to prevent predators from knocking out prey  or static grains
        self.grain_pred_col = self.space.add_collision_handler(4, 5)
        self.grain_pred_col.begin = self.no_collision
        self.prey_pred_col = self.space.add_collision_handler(2, 5)
        self.prey_pred_col.begin = self.no_collision

        # To prevent the differential wall being hit by fish
        self.fish_prey_wall = self.space.add_collision_handler(3, 7)
        self.fish_prey_wall.begin = self.no_collision
        self.fish_prey_wall2 = self.space.add_collision_handler(6, 7)
        self.fish_prey_wall2.begin = self.no_collision
        self.pred_prey_wall2 = self.space.add_collision_handler(5, 7)
        self.pred_prey_wall2.begin = self.no_collision

    def reset(self):
        super().reset()
        self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_size'],
                                                     self.env_variables['width'] - self.env_variables[
                                                         'fish_mouth_size']),
                                   np.random.randint(self.env_variables['fish_mouth_size'],
                                                     self.env_variables['height'] - self.env_variables[
                                                         'fish_mouth_size']))
        self.fish.body.angle = np.random.random() * 2 * np.pi
        self.fish.body.velocity = (0, 0)
        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['width'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120),
                 np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['height'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120)]
                for cloud in range(self.env_variables["prey_cloud_num"])]
            self.build_prey_cloud_walls()

        for i in range(self.env_variables['prey_num']):
            self.create_prey()

        for i in range(self.env_variables['sand_grain_num']):
            self.create_sand_grain()

        for i in range(self.env_variables['vegetation_num']):
            self.create_vegetation()

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None, impulse=None):
        # TODO: Tidy up so is more readable. Do the same with comparable methods in other environment classes.
        self.prey_consumed_this_step = False
        self.last_action = action
        if frame_buffer is None:
            frame_buffer = []
        self.fish.making_capture = True  # TODO: Capture change is here.

        if impulse is not None:
            # To calculate calibration curve.
            reward = self.fish.try_impulse(impulse)
        else:
            reward = self.fish.take_action(action)


        # Add policy helper reward to encourage proximity to prey.
        for ii in range(len(self.prey_bodies)):
            if self.check_proximity(self.prey_bodies[ii].position, self.env_variables['reward_distance']):
                reward += self.env_variables['proximity_reward']

        done = False

        self.fish.hungry += (1 - self.fish.hungry) * self.env_variables['hunger_inc_tau']
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        # TODO: add below to function for clarity.
        if self.predator_shape is None and np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                self.num_steps > self.env_variables['immunity_steps'] and not self.check_fish_near_vegetation():
            self.create_realistic_predator()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey()
            self.displace_sand_grains()
            if self.predator_body is not None:
                self.move_realistic_predator()

            self.space.step(self.env_variables['phys_dt'])
            if self.fish.prey_consumed:
                reward += self.env_variables['capture_basic_reward'] * self.fish.hungry
                self.fish.hungry *= self.env_variables['hunger_dec_tau']
                if len(self.prey_shapes) == 0:
                    done = True
                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False
            if self.fish.touched_predator:
                reward -= self.env_variables['predator_cost']
                done = True
                self.fish.touched_predator = False

            if self.show_all:
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5) / 255.)
                    plt.pause(0.0001)

        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()  # TODO: Needed, but causes index error sometimes.

        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        # Calculate internal state
        in_light = self.fish.body.position[0] > self.dark_col
        if self.env_variables['hunger'] and self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.hungry, self.fish.stress]])
        elif self.env_variables['hunger']:
            internal_state = np.array([[in_light, self.fish.hungry]])
        elif self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.stress]])
        else:
            internal_state = np.array([[in_light]])

        if save_frames or self.draw_screen:
            self.board.erase(bkg=self.env_variables['bkg_scatter'])
            self.draw_shapes()
            self.board.apply_light(self.dark_col, 0.7, 1)
            self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
            self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5) / 255.)
                plt.pause(0.000001)

        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))
        observation = np.clip(observation, 0, 255)

        return observation, reward, internal_state, done, frame_buffer


class Fish:

    """
    Created to simplify the SimState class, while making it easier to have environments with multiple agents in future.
    """
    def __init__(self, board, env_variables, dark_col, realistic_bouts, fish_mass=None):

        # For the purpose of producing a calibration curve.
        if fish_mass is None:
            inertia = pymunk.moment_for_circle(env_variables['fish_mass'], 0, env_variables['fish_head_size'], (0, 0))
        else:
            inertia = pymunk.moment_for_circle(fish_mass, 0, env_variables['fish_mouth_size'], (0, 0))

        self.env_variables = env_variables
        self.body = pymunk.Body(1, inertia)

        self.realistic_bouts = realistic_bouts

        # Mouth
        self.mouth = pymunk.Circle(self.body, env_variables['fish_mouth_size'], offset=(0, 0))
        self.mouth.color = (1, 0, 1)
        self.mouth.elasticity = 1.0
        self.mouth.collision_type = 3

        # Head
        self.head = pymunk.Circle(self.body, env_variables['fish_head_size'], offset=(-env_variables['fish_head_size'], 0))
        self.head.color = (0, 1, 0)
        self.head.elasticity = 1.0
        self.head.collision_type = 6

        # # Tail
        tail_coordinates = ((-env_variables['fish_head_size'], 0),
                            (-env_variables['fish_head_size'], - env_variables['fish_head_size']),
                            (-env_variables['fish_head_size'] - env_variables['fish_tail_length'], 0),
                            (-env_variables['fish_head_size'], env_variables['fish_head_size']))
        self.tail = pymunk.Poly(self.body, tail_coordinates)
        self.tail.color = (0, 1, 0)
        self.tail.elasticity = 1.0
        self.tail.collision_type = 6

        self.verg_angle = env_variables['eyes_verg_angle'] * (np.pi / 180)
        self.retinal_field = env_variables['visual_field'] * (np.pi / 180)
        self.conv_state = 0

        self.left_eye = VisFan(board, self.verg_angle, self.retinal_field, True,
                               env_variables['num_photoreceptors'], env_variables['min_vis_dist'],
                               env_variables['max_vis_dist'], env_variables['dark_gain'],
                               env_variables['light_gain'], env_variables['bkg_scatter'], dark_col)

        self.right_eye = VisFan(board, self.verg_angle, self.retinal_field, False,
                                env_variables['num_photoreceptors'], env_variables['min_vis_dist'],
                                env_variables['max_vis_dist'], env_variables['dark_gain'],
                                env_variables['light_gain'], env_variables['bkg_scatter'], dark_col)

        self.hungry = 0
        self.stress = 1
        self.prey_consumed = False
        self.touched_edge = False
        self.touched_predator = False
        self.making_capture = False
        self.prev_action_impulse = 0

    def take_action(self, action):
        if self.realistic_bouts:
            return self.take_realistic_action(action)
        else:
            return self.take_basic_action(action)

    def take_basic_action(self, action):
        """Original version"""
        if action == 0:  # Swim forward
            reward = -self.env_variables['forward_swim_cost']
            self.body.apply_impulse_at_local_point((self.env_variables['forward_swim_impulse'], 0))
            self.head.color = (0, 1, 0)
        elif action == 1:  # Turn right
            reward = -self.env_variables['routine_turn_cost']
            self.body.angle += self.env_variables['routine_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['routine_turn_impulse'], 0))
            self.head.color = (0, 1, 0)
        elif action == 2:   # Turn left
            reward = -self.env_variables['routine_turn_cost']
            self.body.angle -= self.env_variables['routine_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['routine_turn_impulse'], 0))
            self.head.color = (0, 1, 0)
        elif action == 3:   # Capture
            reward = -self.env_variables['capture_swim_cost']
            self.body.apply_impulse_at_local_point((self.env_variables['capture_swim_impulse'], 0))
            self.head.color = [1, 0, 1]
            self.making_capture = True
        elif action == 4:  # j turn right
            reward = -self.env_variables['j_turn_cost']
            self.body.angle += self.env_variables['j_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['j_turn_impulse'], 0))
            self.head.color = [1, 1, 1]
        elif action == 5:  # j turn left
            reward = -self.env_variables['j_turn_cost']
            self.body.angle -= self.env_variables['j_turn_dir_change']
            self.body.apply_impulse_at_local_point((self.env_variables['j_turn_impulse'], 0))
            self.head.color = [1, 1, 1]
        elif action == 6:   # do nothing:
            reward = -self.env_variables['rest_cost']
        else:
            reward = None
            print("Invalid action given")

        # elif action == 6: #converge eyes. Be sure to update below with fish.[]
        #     self.verg_angle = 77 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 1

        # elif action == 7: #diverge eyes
        #     self.verg_angle = 25 * (np.pi / 180)
        #     self.left_eye.update_angles(self.verg_angle, self.retinal_field, True)
        #     self.right_eye.update_angles(self.verg_angle, self.retinal_field, False)
        #     self.conv_state = 0
        return reward

    def calculate_impulse(self, distance):
        """
        Uses the derived distance-mass-impulse relationship to convert an input distance (in mm) to impulse
        (arbitrary units).
        :param distance:
        :return:
        """
        return (distance*10 - (0.004644*self.env_variables['fish_mass'] + 0.081417))/1.771548

    def calculate_action_cost(self, angle, distance):
        """
        So far, a fairly arbitrary equation to calculate action cost from distance moved and angle changed.
        cost = 0.05(angle change) + 1.5(distance moved)
        :return:
        """
        return abs(angle) * self.env_variables['angle_penalty_scaling_factor'] + (distance**2) * self.env_variables['distance_penalty_scaling_factor']


    def take_realistic_action(self, action):
        if action == 0:  # Slow2
            angle_change, distance = draw_angle_dist(8)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle += np.random.choice([-angle_change, angle_change])
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 1:  # RT right
            angle_change, distance = draw_angle_dist(7)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 2:  # RT left
            angle_change, distance = draw_angle_dist(7)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        elif action == 3:  # Short capture swim
            angle_change, distance = draw_angle_dist(0)
            reward = -self.calculate_action_cost(angle_change, distance) - self.env_variables['capture_swim_extra_cost']
            self.body.angle += np.random.choice([-angle_change, angle_change])
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 1]
            self.making_capture = True

        elif action == 4:  # j turn right
            angle_change, distance = draw_angle_dist(4)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 5:  # j turn left
            angle_change, distance = draw_angle_dist(4)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 1, 1]

        elif action == 6:  # Do nothing
            self.prev_action_impulse = 0
            reward = -self.env_variables['rest_cost']

        elif action == 7:  # c start right
            angle_change, distance = draw_angle_dist(5)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle += angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 0]

        elif action == 8:  # c start left
            angle_change, distance = draw_angle_dist(5)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = [1, 0, 0]

        elif action == 9:  # Approach swim.
            angle_change, distance = draw_angle_dist(10)
            reward = -self.calculate_action_cost(angle_change, distance)
            self.body.angle -= angle_change
            self.prev_action_impulse = self.calculate_impulse(distance)
            self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
            self.head.color = (0, 1, 0)

        else:
            reward = None
            print("Invalid action given")

        return reward

    def try_impulse(self, impulse):
        # Used to produce calibration curve.
        self.body.apply_impulse_at_local_point((impulse, 0))
        return -self.env_variables['j_turn_cost']

    def readings_to_photons(self, readings):
        photons = np.random.poisson(readings * self.env_variables['photon_ratio'])
        if self.env_variables['read_noise_sigma'] > 0:
            noise = np.random.randn(readings.shape[0], readings.shape[1]) * self.env_variables['read_noise_sigma']
            photons += noise.astype(int)
            # photons = photons.clip(0, 255)
        return photons

    def get_visual_inputs(self):
        left_photons = self.readings_to_photons(self.left_eye.readings)
        right_photons = self.readings_to_photons(self.right_eye.readings)
        left_eye = resize(np.reshape(left_photons, (1, len(self.left_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        right_eye = resize(np.reshape(right_photons, (1, len(self.right_eye.vis_angles), 3)) * (255 / self.env_variables['photon_ratio']), (20, self.env_variables['width'] / 2 - 50))
        eyes = np.hstack((left_eye, np.zeros((20, 100, 3)), right_eye))
        eyes[eyes < 0] = 0
        eyes[eyes > 255] = 255
        return eyes


class VisFan:

    def __init__(self, board, verg_angle, retinal_field, is_left, num_arms, min_distance, max_distance, dark_gain,
                 light_gain, bkg_scatter, dark_col):
        self.num_arms = num_arms
        self.distances = np.array([min_distance, max_distance])

        self.vis_angles = None
        self.dist = None
        self.theta = None

        self.update_angles(verg_angle, retinal_field, is_left)
        self.readings = np.zeros((num_arms, 3), 'int')
        self.board = board
        self.dark_gain = dark_gain
        self.light_gain = light_gain
        self.bkg_scatter = bkg_scatter
        self.dark_col = dark_col

        self.width, self.height = self.board.get_size()

    def cartesian(self, bx, by, bangle):
        x = bx + self.dist * np.cos(self.theta + bangle)
        y = (by + self.dist * np.sin(self.theta + bangle))
        return x, y

    def update_angles(self, verg_angle, retinal_field, is_left):
        if is_left:
            min_angle = -np.pi / 2 - retinal_field / 2 + verg_angle / 2
            max_angle = -np.pi / 2 + retinal_field / 2 + verg_angle / 2
        else:
            min_angle = np.pi / 2 - retinal_field / 2 - verg_angle / 2
            max_angle = np.pi / 2 + retinal_field / 2 - verg_angle / 2
        self.vis_angles = np.linspace(min_angle, max_angle, self.num_arms)
        self.dist, self.theta = np.meshgrid(self.distances, self.vis_angles)

    def show_points(self, bx, by, bangle):
        x, y = self.cartesian(bx, by, bangle)
        x = x.astype(int)
        y = y.astype(int)
        for arm in range(x.shape[0]):
            for pnt in range(x.shape[1]):
                if not (x[arm, pnt] < 0 or x[arm, pnt] >= self.width or y[arm, pnt] < 0 or y[arm, pnt] >= self.height):
                    self.board.db[y[arm, pnt], x[arm, pnt], :] = (1, 1, 1)

        [rr, cc] = line(y[0, 0], x[0, 0], y[0, 1], x[0, 1])
        good_points = np.logical_and.reduce((rr > 0, rr < self.height, cc > 0, cc < self.width))
        self.board.db[rr[good_points], cc[good_points]] = (1, 1, 1)
        [rr, cc] = line(y[-1, 0], x[-1, 0], y[-1, 1], x[-1, 1])
        good_points = np.logical_and.reduce((rr > 0, rr < self.height, cc > 0, cc < self.width))
        self.board.db[rr[good_points], cc[good_points]] = (1, 1, 1)

    def read(self, bx, by, bangle):
        x, y = self.cartesian(bx, by, bangle)
        self.readings = self.board.read_rays(x, y, self.dark_gain, self.light_gain, self.bkg_scatter, self.dark_col)


class ContinuousFish(Fish):

    def __init__(self, board, env_variables, dark_col, realistic_bouts, fish_mass=None):
        super().__init__(board, env_variables, dark_col, realistic_bouts, fish_mass)

        self.making_capture = True

    def calculate_distance(self, impulse):
        return (1.771548 * impulse + self.env_variables['fish_mass'] * 0.004644 + 0.081417)/10

    def take_action(self, action):
        impulse = action[0]
        angle = action[1]
        distance = self.calculate_distance(impulse)
        reward = - self.calculate_action_cost(angle, distance) - self.env_variables['baseline_penalty']
        self.prev_action_impulse = impulse
        self.body.apply_impulse_at_local_point((self.prev_action_impulse, 0))
        self.body.angle += angle
        return reward


class NaturalisticEnvironment(BaseEnvironment):

    def __init__(self, env_variables, realistic_bouts, draw_screen=False, fish_mass=None, collisions=True):
        super().__init__(env_variables, draw_screen)

    def reset(self):
        super().reset()
        self.fish.body.position = (np.random.randint(self.env_variables['fish_mouth_size'],
                                                     self.env_variables['width'] - self.env_variables[
                                                         'fish_mouth_size']),
                                   np.random.randint(self.env_variables['fish_mouth_size'],
                                                     self.env_variables['height'] - self.env_variables[
                                                         'fish_mouth_size']))
        self.fish.body.angle = np.random.random() * 2 * np.pi
        self.fish.body.velocity = (0, 0)
        if self.env_variables["differential_prey"]:
            self.prey_cloud_locations = [
                [np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['width'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120),
                 np.random.randint(low=120 + self.env_variables['prey_size'] + self.env_variables['fish_mouth_size'],
                                   high=self.env_variables['height'] - (
                                           self.env_variables['prey_size'] + self.env_variables[
                                       'fish_mouth_size']) - 120)]
                for cloud in range(self.env_variables["prey_cloud_num"])]
            self.build_prey_cloud_walls()

        for i in range(self.env_variables['prey_num']):
            self.create_prey()

        for i in range(self.env_variables['sand_grain_num']):
            self.create_sand_grain()

        for i in range(self.env_variables['vegetation_num']):
            self.create_vegetation()

    def simulation_step(self, action, save_frames, frame_buffer, activations, impulse):

        self.prey_consumed_this_step = False
        self.last_action = action
        if frame_buffer is None:
            frame_buffer = []

        if impulse is not None:
            # To calculate calibration curve.
            reward = self.fish.try_impulse(impulse)
        else:
            reward = self.fish.take_action(action)

        # Add policy helper reward to encourage proximity to prey.
        for ii in range(len(self.prey_bodies)):
            if self.check_proximity(self.prey_bodies[ii].position, self.env_variables['reward_distance']):
                reward += self.env_variables['proximity_reward']

        done = False

        self.fish.hungry += (1 - self.fish.hungry) * self.env_variables['hunger_inc_tau']
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        # TODO: add below to function for clarity.
        if self.predator_shape is None and np.random.rand(1) < self.env_variables["probability_of_predator"] and \
                self.num_steps > self.env_variables['immunity_steps'] and not self.check_fish_near_vegetation():
            self.create_realistic_predator()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.move_prey()
            self.displace_sand_grains()
            if self.predator_body is not None:
                self.move_realistic_predator()

            self.space.step(self.env_variables['phys_dt'])
            if self.fish.prey_consumed:
                reward += self.env_variables['capture_basic_reward'] * self.fish.hungry
                self.fish.hungry *= self.env_variables['hunger_dec_tau']
                if len(self.prey_shapes) == 0:
                    done = True
                self.fish.prey_consumed = False
            if self.fish.touched_edge:
                self.fish.touched_edge = False
            if self.fish.touched_predator:
                reward -= self.env_variables['predator_cost']
                done = True
                self.fish.touched_predator = False

            if self.show_all:
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5) / 255.)
                    plt.pause(0.0001)

        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()  # TODO: Needed, but causes index error sometimes.

        right_eye_pos = (
            -np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            +np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (
            +np.cos(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
            -np.sin(np.pi / 2 - self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        # Calculate internal state
        in_light = self.fish.body.position[0] > self.dark_col
        if self.env_variables['hunger'] and self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.hungry, self.fish.stress]])
        elif self.env_variables['hunger']:
            internal_state = np.array([[in_light, self.fish.hungry]])
        elif self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.stress]])
        else:
            internal_state = np.array([[in_light]])

        if save_frames or self.draw_screen:
            self.board.erase(bkg=self.env_variables['bkg_scatter'])
            self.draw_shapes()
            self.board.apply_light(self.dark_col, 0.7, 1)
            self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
            self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5) / 255.)
                plt.pause(0.000001)

        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))

        return observation, reward, internal_state, done, frame_buffer


class DiscreteNaturalisticEnvironment(NaturalisticEnvironment):

    def __init__(self, env_variables, realistic_bouts, draw_screen=False, fish_mass=None, collisions=True):

        super().__init__(env_variables, realistic_bouts, draw_screen, fish_mass, collisions)

        # Create the fish class instance and add to the space.
        if fish_mass is None:
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts)
        else:
            # In the event that I am producing a calibration curve for distance moved.
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts, fish_mass=fish_mass)

        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)

        # Create walls.
        self.create_walls()
        self.reset()

        self.col = self.space.add_collision_handler(2, 3)
        self.col.begin = self.touch_prey

        if collisions:
            self.pred_col = self.space.add_collision_handler(5, 3)
            self.pred_col.begin = self.touch_predator

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge

        self.edge_pred_col = self.space.add_collision_handler(1, 5)
        self.edge_pred_col.begin = self.remove_realistic_predator

        self.grain_fish_col = self.space.add_collision_handler(3, 4)
        self.grain_fish_col.begin = self.touch_grain

        # to prevent predators from knocking out prey  or static grains
        self.grain_pred_col = self.space.add_collision_handler(4, 5)
        self.grain_pred_col.begin = self.no_collision
        self.prey_pred_col = self.space.add_collision_handler(2, 5)
        self.prey_pred_col.begin = self.no_collision

        # To prevent the differential wall being hit by fish
        self.fish_prey_wall = self.space.add_collision_handler(3, 7)
        self.fish_prey_wall.begin = self.no_collision
        self.fish_prey_wall2 = self.space.add_collision_handler(6, 7)
        self.fish_prey_wall2.begin = self.no_collision
        self.pred_prey_wall2 = self.space.add_collision_handler(5, 7)
        self.pred_prey_wall2.begin = self.no_collision

    def reset(self):
        super().reset()

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None, impulse=None):
        self.fish.making_capture = True  # TODO: Change back
        return super().simulation_step(action, save_frames, frame_buffer, activations, impulse)

