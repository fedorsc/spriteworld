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

import os

import numpy as np

from spriteworld import factor_distributions as distribs
from spriteworld import relational_action_spaces as ra_space
from spriteworld import sprite_generators
from spriteworld import tasks
from spriteworld.configs.cobra import common


def get_config(mode='train'):
    del mode
    task = tasks.FindGoalPosition()

    """
    ============== configure objects & locks ============== 
    """

    possible_shapes = ['square', 'triangle', 'circle']
    max_locks = 2
    max_obj_total = 5
    obj_size = 0.13

    # change rng to a custom one for seeding
    def num_locks(rng=np.random.RandomState()): return rng.randint(0, max_locks + 1)

    def num_sprites(rng=np.random.RandomState()): return rng.randint(1, max_obj_total - max_locks)

    common_sprite_factors = [
        distribs.Continuous('x', 0.1, 0.9),
        distribs.Continuous('y', 0.1, 0.9),
        distribs.Discrete('shape', possible_shapes),
        distribs.Discrete('scale', [obj_size]),
    ]

    ordinary_sprite_factors = [
        distribs.Continuous('c0', 0.2, 0.8),
        distribs.Continuous('c1', 0.3, 1.),
        distribs.Continuous('c2', 0.9, 1.),
    ]
    lock_factors = [
        distribs.Continuous('c0', 1., 1.),
        distribs.Continuous('c1', 1., 1.),
        distribs.Continuous('c2', 1., 1.),
    ]

    lock_gen = sprite_generators.generate_sprites(
        distribs.Product(common_sprite_factors + lock_factors),
        num_sprites=num_locks
    )
    ordinary_gen = sprite_generators.generate_sprites(
        distribs.Product(common_sprite_factors + ordinary_sprite_factors),
        num_sprites=num_sprites
    )

    sprite_gen = sprite_generators.chain_generators(lock_gen, ordinary_gen)

    """
    ============== configure action space ============== 
    """

    act_space = ra_space.MovingAndClicking(
        shapes=possible_shapes,
        object_size=obj_size,
        move_action=ra_space.move_if_unlocked,
        click_action=ra_space.cycle_sprite_shape_if_not_blocked,
        pass_info=False,
        scale=1.0,
        motion_cost=0.1,
        noise_scale=None
    )

    env_config = {
        'task': task,
        'action_space': act_space,
        'renderers': common.renderers(),
        'init_sprites': sprite_gen,
        'max_episode_length': 10,
        'metadata': {
            'name': os.path.basename(__file__)
        }
    }

    return env_config
