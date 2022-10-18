# Copyright 2021 Thomas Schnürer (thomas.schnuerer@tu-ilmenau.de).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# python2 python3

import numpy as np
from absl import logging
from spriteworld.action_spaces import DragAndDrop, specs

"""
============== Possible Object Actions ============== 
"""

class ConditionedMechanic:
    def __int__(self, mechanic, condition=None):
        self.mechanic = mechanic
        self.condition = condition

    def __call__(self, sprite, scene, param, action_space):
        if self.condition(sprite, scene, action_space):
            self.mechanic(sprite, scene, )


def move_sprite_to_position(sprite, scene, motion, action_space):
    motion = sprite.position - goal_pos
    move_sprite(sprite, scene, motion, None)

def move_sprite(sprite, scene, motion, action_space):
    sprite.move(motion, keep_in_frame=True)

def move_if_unlocked(sprite, scene, motion, action_space):
    if is_lock(sprite):
        info = {'is_lock': True}
        motion *= 0  # locks are not allowed to move
    else:
        info = {'is_lock': False}
        if is_locked(sprite, scene):
            logging.info('object locked')
            info['locked'] = True
            motion *= 0
        else:
            info['locked'] = False

    return motion, info


def cycle_sprite_shape_if_not_blocked(sprite, scene, motion, action_space):
    motion *= 0  # just change shape. Don't move.
    if is_blocked(sprite, scene, action_space.object_size):
        logging.info('object blocked')
        info = {'blocked': True}
    else:
        motion, info = cycle_sprite_shape(sprite, scene, motion, action_space)
        info['blocked'] = False

    return motion, info


def cycle_sprite_shape(sprite, scene, motion, action_space):
    """Changes the shape of one sprite to the next one in a list of shapes

    Args:
        sprite: The shape of this sprite will be changed
        scene: List of all sprites currently in the scene
        motion: original motion; will be overwritten to be 0
        action_space: The current action space from which to get the list of shapes
    """
    motion *= 0  # don't move, just change shape
    shapes = action_space.shapes
    assert sprite.shape in shapes, "Shape " + str(sprite.shape) + "not in list of possible shapes. " + str(shapes)

    idx = shapes.index(sprite.shape)
    idx += 1
    if idx >= len(shapes):
        idx = 0
    sprite.shape = shapes[idx]

    return motion, {}


"""
============== Action Space ============== 
"""


def get_sprite_from_position(position, sprites):
    for sprite in sprites[::-1]:
        if sprite.contains_point(position):
            return sprite
    return None


class SymbolicInteraction(object):
    def __init__(
            self,
            shapes,
            object_size,
            interactions_available,
            interactions_costs,
            invalid_action_cost=0.0,
            step_cost=0.0
    ):
        self.shapes = shapes
        self.object_size = object_size
        self.interactions_available = interactions_available
        self.interactions_costs = interactions_costs
        self.invalid_action_cost = invalid_action_cost
        self.step_cost = step_cost
        # self._action_spec = specs.BoundedArray(
        #     shape=(4,), dtype=np.float32, minimum=0.0, maximum=1.0)

    def is_valid_action(self, action):
        return len(action) >= 2 and action[0] >= 0 and action[1] >= 0

    def step(self, action, sprites, keep_in_frame):
        """ Take an action and either click or move a sprite.

        Args:
          action: tuple of (int, int, params (optional) )
            action[0]: index of selected sprite
            action[1]: index of action to perform
            action[2]: optional parameters for action
          sprites: Iterable of sprite.Sprite() instances. If a sprite is moved by
            the action, its position is updated.
          keep_in_frame: Bool. Whether to force sprites to stay in the frame by
            clipping their centers of mass to be in [0, 1].

        Returns:
          Scalar cost of taking this action (if pass_info is set to False) or
          Dict with reward (cost) and info (if pass_info is set to True)
        """

        reward = -self.step_cost

        if not self.is_valid_action(action):
            reward += -self.invalid_action_cost
            return reward

        sprite_idx = action[0]
        action_idx = action[1]
        params = action[2] if len(action) > 2 else []

        if sprite_idx < len(sprites):
            sprite = sprites[sprite_idx]
            if action_idx < len(self.interactions_available) and self.interactions_available[action_idx] is not None:
                motion, info = self.interactions_available[action_idx](sprite, sprites, params, self)
                sprite.move(motion, keep_in_frame=keep_in_frame)
                reward -= self.interactions_costs[action_idx]
            else:
                reward -= self.invalid_action_cost
        else:
            reward -= self.invalid_action_cost

        return reward


class TwoClick(SymbolicInteraction):

    def __init__(
            self,
            shapes,
            object_size,
            move_action=move_if_unlocked,
            click_action=cycle_sprite_shape,
            scale=1.0,
            motion_cost=0.0,
            click_cost=0.0,
            step_cost=0.0,
            noise_scale=None,
            **kwargs
    ):

        super(TwoClick, self).__init__(
            shapes,
            object_size,
            interactions_available=[
                move_action,
                click_action
            ],
            interactions_costs=[
                motion_cost,
                click_cost,
            ],
            step_cost=step_cost,
            **kwargs
        )
        self._scale = scale
        self._action_spec = specs.BoundedArray(
            shape=(4,), dtype=np.float32, minimum=0.0, maximum=1.0)

    def get_motion(self, action):
        pos = action[:2]
        target = action[2:]
        delta_pos = (target - pos) * self._scale
        return delta_pos

    def action_spec(self):
        return self._action_spec

    def step(self, action, sprites, keep_in_frame):
        """ Take an action and either click or move a sprite.

        Args:
          action: Numpy array of shape (4,) in [0, 1]. First two components are the
            position selection, second two are the motion selection.
          sprites: Iterable of sprite.Sprite() instances. If a sprite is moved by
            the action, its position is updated.
          keep_in_frame: Bool. Whether to force sprites to stay in the frame by
            clipping their centers of mass to be in [0, 1].

        Returns:
          Scalar cost of taking this action (if pass_info is set to False) or
          Dict with reward (cost) and info (if pass_info is set to True)
        """
        position_1 = action[:2]
        motion = self.get_motion(action)
        position_2 = position_1 + motion
        sprite_pos_1 = get_sprite_from_position(position_1, sprites)
        sprite_pos_2 = get_sprite_from_position(position_2, sprites)

        if sprite_pos_1 is None:
            sprite_idx = -1

        else:
            sprite_idx = sprites.index(sprite_pos_1)
            if sprite_pos_2 is not None and sprite_pos_2 == sprite_pos_1:
                # action is clicking on an object
                act_idx = 1
            else:
                # action is moving an object
                act_idx = 0

        sym_act = [sprite_idx, act_idx, motion]
        return super(TwoClick, self).step(sym_act, sprites, keep_in_frame)


class MovingAndClicking(DragAndDrop):
    """ This action space takes different actions when an object is clicked or moved (dragged)

    This action space takes in a continuous vector of length 4 with each component
     in [0, 1]. This can be intuited as representing two consecutive clicks:
    [first_x, first_y, second_x, second_y].

    An action is considered as clicking if the distance between the two clicks is below
    a threshold.
    The specific actions to take for clicking and moving can be configured. By default,
    clicking an object will change its shape and dragging it will change its position.
    """

    def __init__(
            self,
            shapes,
            object_size,
            move_action=move_if_unlocked,
            click_action=cycle_sprite_shape,
            pass_info=True,
            scale=1.0,
            motion_cost=0.0,
            click_cost=0.0,
            step_cost=0.0,
            noise_scale=None
    ):
        """ Constructor

        Args:
            shapes: All possible shapes s Sprite could take in the current environment configuration
            move_action(clicked_sprite, sprites, motion, self(action_space)): A function that is called when a
                moving action is performed. Should return motion( = [dx, dy]), ifo ( = dict).
            click_action(clicked_sprite, sprites, motion, self(action_space)): A function that is called when a
                clicking action is performed. Should return motion( = [dx, dy]), ifo ( = dict).
            click_threshold: Determines whether an action is clicking or moving, based on the distance between clicks
            pass_info: If true, step(...) will return a dict with the reward and additional info
                for debugging or analysis of semantic states & interactions
            scale: Multiplier by which the motion is scaled down. Should be in [0.0,1.0].
            motion_cost: Factor by which motion incurs cost.
            noise_scale: Optional stddev of the noise. If scalar, applied to all
                action space components. If vector, must have same shape as action.
        """
        super(MovingAndClicking, self).__init__(scale, motion_cost, noise_scale)

        self.shapes = shapes
        self.object_size = object_size
        self.move_action = move_action
        self.click_action = click_action
        self.pass_info = pass_info
        self.click_cost = click_cost
        self.step_cost = step_cost

    def step(self, action, sprites, keep_in_frame):
        """ Take an action and either click or move a sprite.

        Args:
          action: Numpy array of shape (4,) in [0, 1]. First two components are the
            position selection, second two are the motion selection.
          sprites: Iterable of sprite.Sprite() instances. If a sprite is moved by
            the action, its position is updated.
          keep_in_frame: Bool. Whether to force sprites to stay in the frame by
            clipping their centers of mass to be in [0, 1].

        Returns:
          Scalar cost of taking this action (if pass_info is set to False) or
          Dict with reward (cost) and info (if pass_info is set to True)
        """
        noised_action = self.apply_noise_to_action(action)
        position_1 = noised_action[:2]
        motion = self.get_motion(noised_action)
        position_2 = position_1 + motion
        sprite_pos_1 = self.get_sprite_from_position(position_1, sprites)
        sprite_pos_2 = self.get_sprite_from_position(position_2, sprites)

        reward = -self.step_cost

        if sprite_pos_1 is None:
            reward += 0.
            info = {'selected': -1}

        else:
            info = {'selected': sprites.index(sprite_pos_1)}
            # action is clicking on an object
            if sprite_pos_2 is not None and sprite_pos_2 == sprite_pos_1:
                logging.info('click action')
                info['click'] = True
                motion, inf = self.click_action(sprite_pos_1, sprites, motion, self)
                reward += -self.click_cost

            # action is moving an object
            else:
                logging.info('drag action')
                info['click'] = False
                motion, inf = self.move_action(sprite_pos_1, sprites, motion, self)
                reward += -self._motion_cost * np.linalg.norm(motion)

            sprite_pos_1.move(motion, keep_in_frame=keep_in_frame)
            info = {**info, **inf}

        if self.pass_info:
            return {'reward': reward, 'info': info}
        else:
            return reward


"""
============== Possible Semantic Object States ==============
"""


def is_lock(sprite):
    if sprite.color == (1, 1, 1):
        return True
    return False


def is_locked(sprite, scene):
    """Checks if a sprite is locked by a lock in the scene.
       A sprite will be locked if there is a lock present with the same shape as the sprite
    Args:
        sprite: The sprite to check
        scene: All other sprites in the scene
    Returns:
        Boolean: True if the sprite is locked, False otherwise
    """
    for spr in scene:
        if not is_lock(spr):
            continue
        elif sprite.shape == spr.shape:  # spr is a lock. Has it the same shape as our sprite?
            return True  # Yes! There is a least one lock with the same shape --> sprite is locked
    return False


def is_blocked(sprite, scene, object_size):
    return is_touching_any_sprite(sprite, scene, 1.15 * object_size)


def is_touching_any_sprite(sprite, scene, distance=0.15):
    """Checks if a sprite is touching any other sprite
    Args:
        sprite: The sprite to check
        scene: All other sprites in the scene
        distance: Distance between two sprites below which it is considered blocked
    Returns:
        Boolean: True if sprite is touching any other sprite, False otherwise
    """
    for spr in scene:
        if spr == sprite:
            continue
        dist = np.linalg.norm(sprite.position - spr.position)
        if dist < distance:
            return True

    return False
