# Copyright 2019 DeepMind Technologies Limited.
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
"""Constants for shapes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import numpy as np
from spriteworld import shapes

# A selection of simple shapes
SHAPES = {
    'triangle': shapes.polygon(num_sides=3, theta_0=np.pi/2),
    'square': shapes.polygon(num_sides=4, theta_0=np.pi/4),
    'pentagon': shapes.polygon(num_sides=5, theta_0=np.pi/2),
    'hexagon': shapes.polygon(num_sides=6),
    'octagon': shapes.polygon(num_sides=8),
    'circle': shapes.polygon(num_sides=30),
    'star_4': shapes.star(num_sides=4, theta_0=np.pi/4),
    'star_5': shapes.star(num_sides=5, theta_0=np.pi + np.pi/10),
    'star_6': shapes.star(num_sides=6),
    'spoke_4': shapes.spokes(num_sides=4, theta_0=np.pi/4),
    'spoke_5': shapes.spokes(num_sides=5, theta_0=np.pi + np.pi/10),
    'spoke_6': shapes.spokes(num_sides=6),

    's13': shapes.polygon(num_sides=30),
    's14': shapes.polygon(num_sides=30),
    's15': shapes.polygon(num_sides=30),
    's16': shapes.polygon(num_sides=30),
    's17': shapes.polygon(num_sides=30),
    's18': shapes.polygon(num_sides=30),
    's19': shapes.polygon(num_sides=30),
    's20': shapes.polygon(num_sides=30),
    's21': shapes.polygon(num_sides=30),
    's22': shapes.polygon(num_sides=30),
    's23': shapes.polygon(num_sides=30),
    's24': shapes.polygon(num_sides=30),
    's25': shapes.polygon(num_sides=30),
    's26': shapes.polygon(num_sides=30),
    's27': shapes.polygon(num_sides=30),
}


class ShapeType(enum.IntEnum):
  """Enumerate SHAPES, useful for a state description of the environment."""
  triangle = 1
  square = 2
  pentagon = 3
  hexagon = 4
  octagon = 5
  circle = 6
  star_4 = 7
  star_5 = 8
  star_6 = 9
  spoke_4 = 10
  spoke_5 = 11
  spoke_6 = 12
  #++++++++
  s13 = 13
  s14 = 14
  s15 = 15
  s16 = 16
  s17 = 17
  s18 = 18
  s19 = 19
  s20 = 20
  s21 = 21
  s22 = 22
  s23 = 23
  s24 = 24
  s25 = 25
  s26 = 26
  s27 = 27



