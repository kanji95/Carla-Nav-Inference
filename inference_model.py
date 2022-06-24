#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

# CarlaUE4.exe -windowed -carla-server -quality-level=Low

from __future__ import print_function
from einops import rearrange
import shutil

import matplotlib

from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from carla import ColorConverter as cc
import carla

import wandb

import argparse
import collections
import datetime
import glob
import time
import logging
import math
import os
from pickletools import pyfloat
import numpy.random as random
import re
import sys
import weakref
from pprint import pprint

import numpy as np
from PIL import Image

from sklearn.metrics import pairwise_distances


import torch
import timm
from timesformer.models.vit import VisionTransformer
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter

from models.model import *
from dataloader.word_utils import Corpus
import clip

from solver import Solver


import cv2
from skimage import measure
from skimage.morphology import skeletonize
from scipy import spatial
import queue
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame import K_d
    from pygame import K_z
    from pygame import K_r
    from pygame import K_i
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters)
               if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = True
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        print(self.world.get_blueprint_library().filter(self._actor_filter))
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            if args.infer_dataset == 'val':
                corresponding_maps = ['Town05', 'Town05', 'Town01', 'Town02', 'Town05', 'Town05',
                                      'Town03', 'Town10HD', 'Town02', 'Town05', 'Town05', 'Town10HD',
                                      'Town01', 'Town10HD', 'Town01', 'Town03', 'Town10HD', 'Town07',
                                      'Town03', 'Town01', 'Town05', 'Town03', 'Town10HD', 'Town02',
                                      'Town05']
                other_spawns = [[-9.26923599e+01,  8.44226074e+01,
                                 3.00402393e-01, -1.77731754e+02],
                                [1.17872810e+01, -8.78479691e+01,
                                    3.00346870e-01, -7.66220926e+00],
                                [1.56697952e+02,  3.26596375e+02,
                                    3.00384864e-01, 1.79854262e+02],
                                [1.78018295e+02,  1.05297661e+02,
                                    5.20497061e-01, 1.76370477e+02],
                                [-1.84306503e+02,  6.02764359e+01,
                                    3.55839919e-01, -7.49027179e+01],
                                [-1.84570938e+02, -3.31036072e+01,
                                    3.55855808e-01, -9.14007581e+01],
                                [-8.83062210e+01,  2.15305977e+01,
                                    3.34075755e-01, 8.98495000e+01],
                                [4.47318611e+01,  1.30585266e+02,
                                    3.00308589e-01, -1.78687297e+02],
                                [-3.67999959e+00,  1.21209999e+02,
                                    5.19913651e-01, -8.99998181e+01],
                                [-1.20849388e+02, -1.22122986e+02,
                                    3.00467490e-01, -8.92562086e+01],
                                [-1.25969410e+01, -8.43987503e+01,
                                    3.00346832e-01, 6.38101506e-01],
                                [1.41300917e+01,  6.97140045e+01,
                                    2.99815845e-01, 7.32729899e-02],
                                [3.92337128e+02,  2.87440125e+02,
                                    3.00336990e-01, 8.99912667e+01],
                                [-1.07596748e+02,  7.16208191e+01,
                                    3.00389309e-01, -9.21751170e+01],
                                [1.17757645e+02, -2.05576158e+00,
                                    3.00134048e-01, -1.79646625e+02],
                                [-1.49130745e+01, -1.38322464e+02,
                                    3.00254897e-01, 1.71650645e+02],
                                [9.37591476e+01,  1.29825363e+02,
                                    3.00583935e-01, -3.45377405e+01],
                                [-9.96025162e+01,  4.37102280e+01,
                                    2.99886169e-01, -8.83309023e+01],
                                [-8.85179214e+01, -5.60795135e+01, -
                                    3.65182183e-01, 8.98412037e+01],
                                [8.83797836e+01,  9.92208862e+01,
                                    3.00323562e-01, 8.98171925e+01],
                                [-1.20887642e+02,  3.76235085e+01,
                                    3.00598297e-01, -9.05027520e+01],
                                [-1.22780319e+02, -1.35758575e+02,
                                    9.99672331e-01, 1.62061530e+02],
                                [6.04843826e+01,  1.41189728e+02,
                                    3.00528278e-01, 3.05010811e-01],
                                [1.78290054e+02,  3.02570007e+02,
                                    5.19960381e-01, -1.79999636e+02],
                                [-1.84589813e+02, -7.04510651e+01,
                                    3.55607432e-01, -9.01640295e+01]]
            elif args.infer_dataset == 'test':
                corresponding_maps = ['Town05', 'Town03', 'Town10HD', 'Town01', 'Town05', 'Town03',
                                      'Town02', 'Town03', 'Town05', 'Town02', 'Town05', 'Town05',
                                      'Town05', 'Town01', 'Town01', 'Town10HD', 'Town02', 'Town05',
                                      'Town05', 'Town03', 'Town07', 'Town03', 'Town05', 'Town05',
                                      'Town01', 'Town01', 'Town01', 'Town10HD', 'Town02', 'Town01',
                                      'Town01', 'Town10HD', 'Town02', 'Town05']

                other_spawns = [[1.51480820e+02, -4.27365417e+01,  3.39031866e-01, 8.97952266e+01],
                                [-9.44684219e+00,  1.41458786e+02,
                                    3.00416851e-01, 8.95683141e+01],
                                [-4.18338623e+01, -1.65551643e+01,
                                    3.00070247e-01, -9.01611881e+01],
                                [3.92473267e+02,  2.30671387e+01,
                                    3.00345726e-01, 8.50064448e+01],
                                [-1.20810310e+02, -1.24561485e+02,
                                    3.00484276e-01, -8.10923244e+01],
                                [5.63164234e+00,  1.53826477e+02,
                                    3.00356674e-01, -9.05994198e+01],
                                [1.76574783e+02,  2.40945877e+02,
                                    5.20440914e-01, 6.86590549e-01],
                                [1.10706512e+02, -6.82702827e+00,
                                    3.00383205e-01, -1.79121001e+02],
                                [-1.66296967e+02, -4.21021843e+00,
                                    3.00320625e-01, -1.71883268e+02],
                                [1.46535995e+02,  3.02544861e+02,
                                    5.20397879e-01, 1.79977370e+02],
                                [-1.84470978e+02, -2.71637936e+01,
                                    3.55837420e-01, -9.45089721e+01],
                                [-1.84637863e+02,  1.09748627e+02,
                                    3.56090468e-01, -9.39120116e+01],
                                [-1.25969410e+01, -8.43987503e+01,
                                    3.00346832e-01, 6.38101506e-01],
                                [1.34890442e+02,  1.33428787e+02,
                                    3.00334396e-01, 6.14681041e-01],
                                [8.83620987e+01,  3.43473473e+01,
                                    3.00381927e-01, 9.08729553e+01],
                                [3.18485489e+01,  1.40787888e+02,
                                    3.00343570e-01, -4.29834205e+00],
                                [1.73870056e+02,  1.09400040e+02,
                                    5.20033695e-01, -1.83105474e-04],
                                [-1.46840958e+02, -8.46176071e+01,
                                    3.00092544e-01, -5.13183552e-01],
                                [-1.50300430e+02, -8.43797150e+01,
                                    3.00304661e-01, 1.11999618e+01],
                                [1.50086563e+02, -1.10106041e+02,
                                    8.30025009e+00, 9.14936144e+01],
                                [-5.17965279e+01, -8.82258453e+01,
                                    2.99742241e-01, -8.96606526e-02],
                                [-6.18445063e+00,  1.04491875e+02,
                                    3.04057273e-01, 8.99171581e+01],
                                [-1.84589813e+02, -7.04510651e+01,
                                    3.55607432e-01, -9.01640295e+01],
                                [6.91786041e+01, -1.45469162e+02,
                                    3.38874053e-01, 1.94061631e+00],
                                [-2.06940818e+00,  9.50440121e+00,
                                    2.99929618e-01, 9.09264084e+01],
                                [1.12055565e+02,  1.91608566e+02,
                                    5.20440616e-01, -3.77898660e-01],
                                [1.58059494e+02,  1.83820286e+01,
                                    3.00492706e-01, -8.99103017e+01],
                                [-1.13558014e+02, -1.77364063e+01,
                                    3.00432529e-01, 8.45108810e+01],
                                [1.36133987e+02,  2.08886505e+02,
                                    5.20381831e-01, -9.02495434e+01],
                                [1.09782021e+02,  1.95143631e+02,
                                    3.00322456e-01, -1.79884083e+02],
                                [9.23820648e+01,  7.63002396e+01,
                                    3.00486411e-01, -9.00924628e+01],
                                [-4.16684113e+01,  1.13936279e+02,
                                    3.00354862e-01, -9.98055513e+01],
                                [1.93697357e+02,  2.61185120e+02,
                                    5.20474680e-01, -9.03951562e+01],
                                [5.43945541e+01, -9.14242859e+01,
                                    3.00569495e-01, -1.76808831e+02]]
            elif args.infer_dataset == 'val5':

                corresponding_maps = ['Town10HD', 'Town03',
                                      'Town05', 'Town10HD',
                                      'Town01', 'Town05',
                                      'Town05', 'Town03',
                                      'Town02', 'Town01',
                                      'Town05', 'Town03',
                                      'Town01', 'Town03',
                                      'Town05', 'Town03',
                                      'Town01', 'Town02',
                                      'Town02', 'Town05',
                                      'Town02', 'Town05',
                                      'Town10HD', 'Town03',
                                      'Town01', 'Town01',
                                      'Town07', 'Town05',
                                      'Town02', 'Town10HD',
                                      'Town03', 'Town03',
                                      'Town05', 'Town10HD',
                                      'Town03', 'Town07',
                                      'Town10HD', 'Town05',
                                      'Town05', 'Town05',
                                      'Town01', 'Town10HD',
                                      'Town05', 'Town01',
                                      'Town01', 'Town05',
                                      'Town05', 'Town05',
                                      'Town05', 'Town01']
                ['Town03', 'Town03', 'Town03', 'Town03', 'Town01', 'Town05', 'Town03', 'Town10HD', 'Town05', 'Town05', 'Town10HD', 'Town03',
                 'Town03', 'Town10HD', 'Town03', 'Town10HD', 'Town02', 'Town07', 'Town03', 'Town01', 'Town10HD', 'Town10HD', 'Town01', 'Town10HD', 'Town10HD']
                other_spawns = [[60.48438262939453,
                                141.18972778320312,
                                0.3005282778292894,
                                0.30501081094723576],
                                [150.08656311035156, -110.10604095458984,
                                    8.30025009084493, 91.4936144379134],
                                [-12.596940994262695,
                                -84.39875030517578,
                                0.3003468319773674,
                                0.6381015056073027],
                                [14.130091667175293,
                                69.71400451660156,
                                0.2998158449307084,
                                0.07327298994360518],
                                [134.89044189453125,
                                133.4287872314453,
                                0.30033439602702855,
                                0.6146810410042063],
                                [-184.30650329589844,
                                60.27643585205078,
                                0.35583991948515176,
                                -74.90271786032402],
                                [-184.47097778320312,
                                -27.163793563842773,
                                0.3558374198153615,
                                -94.50897205943477],
                                [-88.30622100830078,
                                21.530597686767578,
                                0.3340757546946406,
                                89.84950003104187],
                                [146.53599548339844,
                                302.54486083984375,
                                0.520397879369557,
                                179.97736976081973],
                                [392.4732666015625, 23.067138671875,
                                0.30034572556614875, 85.00644477467706],
                                [69.17860412597656,
                                -145.4691619873047,
                                0.33887405339628457,
                                1.9406163128470022],
                                [-9.446842193603516,
                                141.4587860107422,
                                0.30041685067117213,
                                89.56831411795777],
                                [117.75764465332031,
                                -2.0557615756988525,
                                0.3001340480521321,
                                -179.64662467034285],
                                [110.70651245117188,
                                -6.827028274536133,
                                0.300383204780519,
                                -179.12100060558794],
                                [-184.58981323242188,
                                -70.45106506347656,
                                0.35560743156820535,
                                -90.16402948723683],
                                [-88.5179214477539,
                                -56.07951354980469,
                                -0.36518218349665404,
                                89.84120365225724],
                                [156.6979522705078,
                                326.59637451171875,
                                0.30038486439734696,
                                179.85426244937062],
                                [-3.679999589920044,
                                121.20999908447266,
                                0.5199136512354017,
                                -89.99981808937414],
                                [173.87005615234375,
                                109.40003967285156,
                                0.5200336949899793,
                                -0.0001831054738447505],
                                [151.48081970214844,
                                -42.736541748046875,
                                0.33903186600655316,
                                89.79522662591198],
                                [178.01829528808594,
                                105.29766082763672,
                                0.5204970614984632,
                                176.37047726606554],
                                [-120.8493881225586,
                                -122.12298583984375,
                                0.3004674904048443,
                                -89.25620863215258],
                                [-107.59674835205078,
                                71.62081909179688,
                                0.300389308668673,
                                -92.17511696926216],
                                [5.6316423416137695,
                                153.82647705078125,
                                0.3003566741943359,
                                -90.59941984702316],
                                [-2.0694081783294678,
                                9.504401206970215,
                                0.29992961809039115,
                                90.92640835629265],
                                [88.36209869384766,
                                34.347347259521484,
                                0.3003819270059466,
                                90.87295525533357],
                                [-51.79652786254883,
                                -88.22584533691406,
                                0.2997422406449914,
                                -0.08966065264642963],
                                [-184.6378631591797,
                                109.74862670898438,
                                0.3560904676094651,
                                -93.91201157373341],
                                [178.29005432128906,
                                302.57000732421875,
                                0.5199603812769056,
                                -179.9996361787483],
                                [31.848548889160156,
                                140.7878875732422,
                                0.3003435704857111,
                                -4.29834204509062],
                                [-122.78031921386719,
                                -135.75857543945312,
                                0.9996723311021924,
                                162.061529676021],
                                [-14.913074493408203,
                                -138.3224639892578,
                                0.3002548974007368,
                                171.65064528036277],
                                [11.787281036376953,
                                -87.84796905517578,
                                0.30034687016159295,
                                -7.662209257643786],
                                [-41.8338623046875,
                                -16.555164337158203,
                                0.3000702468678355,
                                -90.16118813760058],
                                [49.407962799072266,
                                -192.74472045898438,
                                0.30028278306126593,
                                2.552531121288393],
                                [-99.6025161743164,
                                43.710227966308594,
                                0.2998861690983176,
                                -88.33090233275658],
                                [93.75914764404297, 129.8253631591797,
                                0.300583934597671, -34.53774053067005],
                                [-54.064979553222656,
                                35.9079704284668,
                                0.35580652225762605,
                                90.35752191012398],
                                [-120.7091064453125,
                                58.98898696899414,
                                0.3000205412507057,
                                -91.23781705760003],
                                [-92.6923599243164, 84.422607421875,
                                0.3004023928195238, -177.73175370854952],
                                [88.3797836303711, 99.22088623046875,
                                0.30032356195151805, 89.81719249579363],
                                [44.73186111450195,
                                130.58526611328125,
                                0.30030858907848595,
                                -178.687297299735],
                                [-6.284280300140381,
                                -91.40370178222656,
                                0.30040359422564505,
                                -179.89480642577922],
                                [392.3371276855469,
                                287.44012451171875,
                                0.30033698976039885,
                                89.99126669357798],
                                [338.88525390625, 21.121721267700195,
                                0.3003598779439926, -89.97128156148169],
                                [-184.57093811035156,
                                -33.103607177734375,
                                0.3558558078482747,
                                -91.40075811590339],
                                [-150.30043029785156,
                                -84.37971496582031,
                                0.3003046607598662,
                                11.199961794020014],
                                [-12.596940994262695,
                                -84.39875030517578,
                                0.3003468319773674,
                                0.6381015056073027],
                                [-184.58981323242188,
                                -70.45106506347656,
                                0.35560743156820535,
                                -90.16402948723683],
                                [338.88525390625, 21.121721267700195, 0.3003598779439926, -89.97128156148169]]
            if args.spawn == -1:
                spawn_point = random.choice(
                    spawn_points) if spawn_points else carla.Transform()
            else:
                assert args.spawn < len(other_spawns)
                if corresponding_maps[args.spawn] != args.map:
                    print(f'Spawn for {corresponding_maps[args.spawn]}')
                spawn_point = carla.Transform(carla.Location(
                    x=other_spawns[args.spawn][0], y=other_spawns[args.spawn][1], z=other_spawns[args.spawn][2]), carla.Rotation(yaw=other_spawns[args.spawn][3]))
            # Fix Spawning Point
            # spawn_point = spawn_points[0] if spawn_points else carla.Transform(
            # )
            print(f'sample spawn_point: {spawn_points[0].location}')
            print(f'spawn_points: {spawn_point}')

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        global saving
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                self._is_ignore_shortcut(event.key)
                self._is_next_episode_shortcut(event.key)
                self._is_delete_episode_shortcut(event.key)
                self._is_rename_shortcut(event.key)

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    @staticmethod
    def _is_ignore_shortcut(key):
        global saving
        global command_given
        if key == K_i:
            saving = [False, True, False]

    @staticmethod
    def _is_next_episode_shortcut(key):
        global saving
        global command_given
        global mask_video
        global frame_video
        global traj_mask_video
        global target_video
        global full_video
        global target_number
        global pred_found
        global frame_count
        global command
        global num_preds

        if key == K_d:
            command_given = False
            saving = [True, True, False]
            print('Episode Done')
            pred_found = 0
            num_preds = 0

            for full_video_index in range(len(full_video)):
                pass

            frame_video = np.concatenate(frame_video, axis=0)
            full_video = np.concatenate(full_video, axis=0)
            target_video = np.concatenate(target_video, axis=0)
            mask_video = np.concatenate(mask_video, axis=0)
            traj_mask_video = np.concatenate(traj_mask_video, axis=0)

            # import pdb; pdb.set_trace()
            mask_video_overlay = np.copy(frame_video)
            mask_video_overlay[:,
                               0] += (mask_video[:, 0]/mask_video.max())
            if mask_video.shape[1] == 2:
                mask_video_overlay[:,
                                   1] += (mask_video[:, 1]/mask_video.max())
            mask_video_overlay = np.clip(
                mask_video_overlay, a_min=0., a_max=1.)

            traj_mask_video_overlay = np.copy(frame_video)
            traj_mask_video_overlay[:,
                                    0] += (traj_mask_video[:, 0]/traj_mask_video.max())
            traj_mask_video_overlay = np.clip(
                traj_mask_video_overlay, a_min=0., a_max=1.)

            frame_video = np.uint8(frame_video * 255)
            full_video = np.uint8(full_video * 255)
            target_video = np.uint8(target_video * 255)
            mask_video = np.uint8(mask_video_overlay * 255)
            traj_mask_video = np.uint8(traj_mask_video_overlay * 255)

            print(full_video.shape, frame_video.shape)

            wandb.log(
                {
                    "full_video": wandb.Video(full_video, fps=10, caption=command, format="mp4"),
                    "video": wandb.Video(frame_video, fps=1, caption=command, format="mp4"),
                    "target video": wandb.Video(target_video, fps=1, caption=command, format="mp4"),
                    "pred_mask": wandb.Video(mask_video, fps=1, caption=command, format="mp4"),
                    "traj_mask": wandb.Video(traj_mask_video, fps=1, caption=command, format="mp4"),
                }
            )
            frame_video = []
            mask_video = []
            target_video = []
            traj_mask_video = []
            full_video = []

    @staticmethod
    def _is_delete_episode_shortcut(key):
        global saving
        global command_given
        if key == K_z:
            command_given = False
            saving[1] = True
            saving[2] = True

    @staticmethod
    def _is_rename_shortcut(key):
        global episode_number
        global command_given
        if key == K_r:
            command = input('Re-enter correct command: ')
            with open(f'_out/{episode_number}/command.txt', 'w') as f:
                f.write(command)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(
                world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 *
                                       math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (
                transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' %
                                (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' %
                            (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        player_position = world.player.get_transform().location
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles),
            f'Location: X={player_position.x:.3f}, Y={player_position.y:.3f}']

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x)
                    for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(
                            display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255),
                                         rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(
                        item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 *
                    self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        print(self._parent)
        self.hud = hud
        self.recording = True
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-0, y=0, z=2), carla.Rotation(pitch=0.0)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                # if "rgb" in item[0]:
                #     blp.set_attribute('fov', '150')
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            print(index, self.transform_index,
                  self._camera_transforms[self.transform_index][0])
            print(self.sensors[index])
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image:
                               camera_manager_listen_event(image, [rgb_cam_queue.put, lambda image: CameraManager._parse_image(weak_self, image)]))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' %
                              ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        global command_given
        global episode_number
        global saving
        global agent
        global depth_camera
        global target_number
        global pred_found
        global frame_count
        global weak_dc
        global weak_agent
        global K
        global destination
        global command
        global world
        global clock

        global network
        global corpus
        global img_transform
        global unnormalized_transform
        global phrase
        global phrase_mask
        global frame_mask
        global threshold
        global confidence

        global frame_video
        global traj_mask_video
        global mask_video
        global target_video

        global frame_pending

        global depth_cam_queue
        global rgb_cam_queue

        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(
                lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording and command_given and saving[0]:
            if 'target_destination' in agent.__dict__ and agent.target_destination is not None:
                os.makedirs(f'_out/{episode_number}', exist_ok=True)
                os.makedirs(f'_out/{episode_number}/images', exist_ok=True)
                os.makedirs(
                    f'_out/{episode_number}/inverse_matrix', exist_ok=True)

                np.save(f'_out/{episode_number}/inverse_matrix/{image.frame:08d}.npy',
                        np.array(image.transform.get_inverse_matrix()))
                img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                img = np.reshape(
                    img, (image.height, image.width, 4))  # RGBA format
                img = img[:, :, :]  # BGR
                # im.save(f'_out/{episode_number}/images/{image.frame:08d}.png')

                cv2.imwrite(
                    f'_out/{episode_number}/images/{image.frame:08d}.png', img)

                # image.save_to_disk(
                #     f'_out/{episode_number}/images/{image.frame:08d}')
                with open(f'_out/{episode_number}/vehicle_positions.txt', 'a+') as f:
                    f.write(
                        f'{agent._vehicle.get_transform().location.x},{agent._vehicle.get_transform().location.y},{agent._vehicle.get_transform().location.z}\n')
                with open(f'_out/{episode_number}/target_positions.txt', 'a+') as f:
                    f.write(
                        f'{agent.target_destination.x},{agent.target_destination.y},{agent.target_destination.z},{target_number}\n')

            frame_pending = 1


def camera_manager_listen_event(image, functions):
    for f in functions:
        f(image)


def make_timeline(past):
    vals = np.array(past)
    x = vals[:, 0]
    y = vals[:, 1]
    z = vals[:, 2]

    first_x = 0
    end_x = int(x.shape[0]/5)

    line_rot = np.arctan((y[end_x]-y[first_x]) /
                         (x[end_x]-x[first_x]+1e-9)) * 180/np.pi

    if x[end_x]-x[first_x] < 0:
        line_rot += 180
    elif x[end_x] == x[first_x] and y[first_x]-y[end_x] < 0:
        line_rot += 180

    rot = matplotlib.transforms.Affine2D().rotate_deg(90-line_rot)

    fig = plt.figure()

    points = vals[:, :3]
    points[:, 2] = 1

    temp_out = rot.transform(points[:, :2])

    x = -temp_out[:, 0]
    y = temp_out[:, 1]

    plt.plot(x, y, color='black')

    y_min = np.min([y[0]-20, np.min(y)-10])
    y_max = np.max([y[0]+80, np.max(y)+10])

    x_size = np.max([50, np.max(x)-x[0]+10, x[0]-np.min(x)+10])
    x_min = x[0]-x_size
    x_max = x[0]+x_size

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal')
    plt.axis('off')

    fig.canvas.draw()
    timeline = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    timeline = timeline.reshape(fig.canvas.get_width_height()[::-1]+(3,))
    plt.close()
    timeline = timeline[:, :, 0].astype(np.float32)/255
    timeline = 1-timeline
    return timeline


def process_network(image, depth_cam_data, vehicle_matrix, vehicle_location, sampling):
    global command_given
    global episode_number
    global saving
    global agent
    global depth_camera
    global target_number
    global frame_count
    global weak_dc
    global weak_agent
    global K
    global destination
    global command
    global world
    global clock

    global network
    global corpus
    global img_transform
    global unnormalized_transform
    global phrase
    global phrase_mask
    global frame_mask
    global threshold
    global confidence

    global frame_video
    global traj_mask_video
    global mask_video
    global target_video

    global frame_pending

    global new_destination

    global video_queue
    global full_video
    global past

    global pred_found
    global num_preds

    global prev_pred
    global prev_preds

    global prev_loc

    global args

    global mode

    # import pdb
    # pdb.set_trace()
    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(
        img, (image.height, image.width, 4))  # RGBA format
    img = img[:, :, :]  # BGR

    im = Image.fromarray(img[:, :, :3][:, :, ::-1])

    frame = img_transform(im)
    un_frame = unnormalized_transform(im)

    full_video.append(un_frame.unsqueeze(0))

    if mode == 'video':
        if frame_count == 0:
            video_queue = [frame]*args.num_frames*args.one_in_n
        else:
            video_queue.pop()
            video_queue.append(frame)

    if frame_count == 0:
        past = []
    past.append([image.transform.location.x,
                image.transform.location.y, image.transform.location.z])

    if frame_count == 0 and target_number == 0:
        frame_video = []
        mask_video = []
        traj_mask_video = []
        target_video = []

        prev_preds = []

    if pred_found or num_preds >= args.num_preds:
        pred_found = 0
        return

    if prev_loc is not None and ((prev_loc.x - vehicle_location.x)**2 + (prev_loc.y - vehicle_location.y)**2)**0.5 < args.min_distance and agent.target_destination is not None:
        return

    if frame_count % sampling == 0:

        frame = frame.cuda(
            non_blocking=True).unsqueeze(0)

        if mode != 'video':
            mask, traj_mask = network(frame, phrase, frame_mask, phrase_mask)
        else:
            video_frames = video_queue[::-1][::args.one_in_n][::-1]
            # video_frames = full_video[-args.num_frames *
            #                           args.one_in_n][::-1][::args.one_in_n][::-1]
            video_frames = torch.stack(video_frames, dim=1).cuda(
                non_blocking=True).unsqueeze(0)
            if 'clip_' in args.img_backbone:
                video_frames = video_frames.float()
                # phrase = phrase.float()
                frame_mask = frame_mask.float()

                timeline = make_timeline(past)
                timeline = torch.Tensor(timeline).cuda(
                    non_blocking=True).unsqueeze(0).float()

            if 'clip_' in args.img_backbone:
                mask, traj_mask = network(
                    video_frames, phrase, frame_mask, phrase_mask
                )
            else:
                mask, traj_mask = network(
                    video_frames, phrase, frame_mask, phrase_mask)

        if len(mask.shape) == 5:
            mask = mask.detach()[:, -1]

        mask_np = mask.detach().cpu().numpy().transpose(2, 3, 1, 0)
        intermediate_mask_np = mask_np[:, :, 0].reshape(
            mask_np.shape[0], mask_np.shape[1])
        if mask_np.shape[2] == 1:
            final_mask_np = mask_np[:, :, 0].reshape(
                mask_np.shape[0], mask_np.shape[1])
        else:
            final_mask_np = mask_np[:, :, 1].reshape(
                mask_np.shape[0], mask_np.shape[1])

        # print(intermediate_mask_np.shape,
        #       intermediate_mask_np.max(), intermediate_mask_np.min())
        # print(final_mask_np.shape, final_mask_np.max(), final_mask_np.min())

        mask_np = intermediate_mask_np+final_mask_np
        mask_np = mask_np/2

        traj_mask_np = traj_mask.detach().cpu()
        traj_mask_np = rearrange(traj_mask_np, "1 1 h w -> h w").numpy()
        traj_mask_np = skeletonize(cv2.threshold(
            traj_mask_np, args.traj_threshold, 1, cv2.THRESH_BINARY)[1])

        # mask_np = cv2.resize(mask_np, (1280, 720))
        if args.target == 'mask' or args.target == 'distance':
            pixel_out = best_pixel(mask_np, threshold, confidence)
            if pixel_out != -1:
                probs, region = pixel_out
                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                   region, K, destination, set_destination=True)
        if args.target == 'mask_dual':
            pixel_out = best_pixel(final_mask_np, threshold, confidence)
            if pixel_out == -1:
                pixel_out = best_pixel(
                    intermediate_mask_np, threshold, confidence)
            if pixel_out != -1:
                probs, region = pixel_out
                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                   region, K, destination, set_destination=True)
        elif args.target == 'trajectory':
            pixel_out = best_pixel_traj(traj_mask_np)
            probs, region = pixel_out
            if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                               region, K, destination, set_destination=True)
        elif args.target == 'network':
            pixel_out = best_pixel(final_mask_np, threshold, confidence)
            if pixel_out == -1:
                pixel_out = best_pixel(
                    intermediate_mask_np, threshold, confidence)
            if pixel_out[0] >= confidence:
                pixel_temp = best_pixel(
                    intermediate_mask_np, threshold, confidence)
                if pixel_temp != -1 and pixel_temp[0] >= pixel_out[0]:
                    pixel_out = pixel_temp
                else:
                    pred_found = 1
            elif pixel_out[0] < confidence:
                pixel_temp = best_pixel(final_mask_np, threshold, confidence)
                pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                               pixel_temp[1], K, destination, set_destination=False)
                blue_dest = new_destination

                pixel_temp = best_pixel_traj(traj_mask_np)
                pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                               pixel_temp[1], K, destination, set_destination=False)
                yellow_dest = new_destination

                distance_blue_traj = np.linalg.norm(
                    np.array([blue_dest.x, blue_dest.y])
                    - np.array([yellow_dest.x, yellow_dest.y]))
                print(
                    f'------------------distance_blue_traj {distance_blue_traj}------------------')
                if distance_blue_traj < args.distance:
                    pixel_out = best_pixel(
                        final_mask_np, threshold, confidence)
                    pred_found = 1
                else:
                    pixel_out = best_pixel_traj(traj_mask_np)
        elif args.target == 'network2':
            pixel_out = best_pixel_traj(traj_mask_np)
            pixel_temp = best_pixel(mask_np, threshold, confidence)
            if pixel_temp != -1:
                pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                               pixel_temp[1], K, destination, set_destination=False)
                blue_dest = new_destination

                pixel_temp = best_pixel_traj(traj_mask_np)
                pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                               pixel_temp[1], K, destination, set_destination=False)
                yellow_dest = new_destination

                distance_blue_traj = np.linalg.norm(
                    np.array([blue_dest.x, blue_dest.y])
                    - np.array([yellow_dest.x, yellow_dest.y]))
                print(
                    f'------------------distance_blue_traj {distance_blue_traj}------------------')
                if distance_blue_traj < args.distance:
                    pixel_out = best_pixel(
                        final_mask_np, threshold, confidence)
                    pred_found = 1
        if pixel_out != -1:
            probs, region = pixel_out
            if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                               region, K, destination, set_destination=True)

        if pixel_out != -1:

            probs, region = pixel_out

            print(
                f'probs = {probs:.2f} with pred_found = {pred_found} and num_preds = {num_preds}')

            color = (255, 0, 0)

            ########### STOPPING CRITERIA START ################
            if args.stop_criteria == 'confidence':
                if args.target == 'mask':
                    pixel_temp = best_pixel(
                        mask_np, threshold=args.threshold, confidence=args.confidence)
                    if pixel_temp != -1:
                        probs, region = pixel_temp
                        print(
                            f"+++++++++++Confidence: {probs}++++++++++++++++++++++")
                        if probs >= confidence:
                            if args.sub_command or True:
                                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                                   region, K, destination, set_destination=True)

                            color = (0, 0, 255)
                            pred_found = 1
                        else:
                            color = (255, 0, 0)
                elif args.target == 'trajectory':
                    pixel_temp = best_pixel(
                        final_mask_np, threshold=args.threshold, confidence=args.confidence)
                    if pixel_temp != -1:
                        probs, region = pixel_temp
                        print(
                            f"+++++++++++Confidence: {probs}++++++++++++++++++++++")
                        if probs >= confidence:
                            if args.sub_command or True:
                                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                                   region, K, destination, set_destination=True)

                            color = (0, 0, 255)
                            pred_found = 1
                        else:
                            color = (255, 0, 0)

                elif args.target == 'mask_dual':
                    pixel_temp = best_pixel(
                        final_mask_np, threshold=args.threshold, confidence=args.confidence)
                    pixel_temp_intermediate = best_pixel(
                        intermediate_mask_np, threshold=args.threshold, confidence=args.confidence)
                    print(
                        f"+++++++++++Final Confidence: {pixel_temp[0] if pixel_temp!=-1 else 'NA'},\
                             Intermediate Confidence: {pixel_temp_intermediate[0] if pixel_temp_intermediate!=-1 else 'NA'}++++++++++++++++++++++")

                    if pixel_temp != -1 and pixel_temp_intermediate != -1 and pixel_temp_intermediate[0] >= pixel_temp[0]:
                        probs, region = pixel_temp_intermediate
                        if probs >= confidence:
                            if args.sub_command or True:
                                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                                   region, K, destination, set_destination=True)
                            color = (255, 0, 0)

                    elif pixel_temp != -1:
                        probs, region = pixel_temp
                        print(
                            f"+++++++++++Confidence: {probs}++++++++++++++++++++++")
                        if probs >= confidence:
                            if args.sub_command or True:
                                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                                   region, K, destination, set_destination=True)

                            color = (0, 0, 255)
                            pred_found = 1
                        else:
                            color = (255, 0, 0)
                else:
                    raise NotImplementedError(
                        f'{args.target} does not work with stop_criteria confidence')

            elif args.stop_criteria == 'distance' and agent.target_destination:
                if args.target == 'trajectory':
                    pixel_temp = best_pixel(mask_np, threshold, confidence)
                    if pixel_temp != -1:
                        pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                       pixel_temp[1], K, destination, set_destination=False)

                        if np.linalg.norm(
                            np.array([vehicle_location.x, vehicle_location.y])
                            - np.array([new_destination.x,
                                        new_destination.y])) < args.distance:

                            color = (0, 0, 255)
                            pred_found = 1

                            if args.sub_command or True:
                                probs, region = pixel_temp
                                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                                   region, K, destination, set_destination=True)

                print(
                    f'Distance from target: {np.linalg.norm(np.array([vehicle_location.x, vehicle_location.y])- np.array([agent.target_destination.x,agent.target_destination.y]))}')
                if np.linalg.norm(
                    np.array([vehicle_location.x, vehicle_location.y])
                    - np.array([agent.target_destination.x,
                                agent.target_destination.y])) < args.distance:

                    color = (0, 0, 255)
                    pred_found = 1
            elif args.stop_criteria == 'consistent':
                temp_pred = 0
                if args.target == 'mask':
                    pixel_temp = best_pixel(
                        mask_np, threshold=args.threshold, confidence=args.min_confidence, method="mean_wa")
                    if pixel_temp != -1:
                        probs, region = pixel_temp
                        print(
                            f"+++++++++++Confidence: {probs}++++++++++++++++++++++")
                        if probs >= args.min_confidence:
                            if args.sub_command or True:
                                if probs > args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                                   region, K, destination, set_destination=True)
                                    prev_preds.append(target_vehicle_dist)

                            color = (0, 0, 255)
                            temp_pred = 1
                        else:
                            color = (255, 0, 0)
                elif args.target == 'distance':
                    pixel_temp = best_pixel(
                        mask_np, threshold=args.threshold, confidence=args.min_confidence, method="mean_wa")
                    if pixel_temp != -1:
                        probs, region = pixel_temp
                        print(
                            f"+++++++++++Confidence: {probs}++++++++++++++++++++++")
                        if probs >= args.min_confidence or probs == -1 or np.random.randint(1000) < args.confidence_probs*100:
                            pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                           region, K, destination, set_destination=True)
                            target_vehicle_dist = np.linalg.norm(np.array([vehicle_location.x, vehicle_location.y])
                                                                 - np.array([agent.target_destination.x, agent.target_destination.y]))
                            if target_vehicle_dist < 1.75 * args.considered_distance:
                                prev_preds.append(agent.target_destination)
                            if target_vehicle_dist < args.considered_distance and len(prev_preds) > args.num_preds:
                                print(
                                    "!!!!!!!!!!!!!!!!!!!!!!IN!!!!!!!!!!!!!!!!!!!!!!!!!")
                                temp_pred = 1
                else:
                    return NotImplementedError(f'{args.target} does not work with stop_criteria consistent')

                if temp_pred:
                    pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                                   region, K, destination, set_destination=False)
                    if args.target == 'mask':
                        if "prev_pred" not in globals():
                            prev_pred = None
                        if prev_pred is not None:
                            distance_bw = np.linalg.norm(np.array([prev_pred.x, prev_pred.y]) -
                                                         np.array([new_destination.x, new_destination.y]))
                            if distance_bw < args.distance:
                                pred_found = 1
                            else:
                                # num_preds = 0
                                pred_found = -1
                        else:
                            pred_found = 1
                        prev_pred = new_destination
                    elif args.target == 'distance':
                        last_points = [[tgt_point.x, tgt_point.y]
                                       for tgt_point in prev_preds[-min(len(prev_preds), int(args.num_preds*1.5)):]]
                        counts_followed = np.sum(pairwise_distances(
                            np.array(last_points)) < args.distance)
                        if counts_followed >= min(4, args.num_preds) * len(last_points):
                            num_preds = args.num_preds+1
                            pred_found = 1

                ########### STOPPING CRITERIA END ################

            probs, region = pixel_out
            print(
                f'----------------------region:{region}----------------------')

            frame_video.append(un_frame.unsqueeze(0).detach().cpu().numpy())
            mask_video.append(mask.detach().cpu().numpy())
            traj_mask_video.append(
                skeletonize(
                    cv2.threshold(
                        traj_mask.detach().cpu().numpy().reshape(traj_mask.shape[2], traj_mask.shape[3]), args.traj_threshold, 1, cv2.THRESH_BINARY)[1]).reshape(
                            1, 1, traj_mask.shape[2], traj_mask.shape[3]))

            # print(un_frame.unsqueeze(0).shape)
            target_vector = np.copy(
                un_frame.unsqueeze(0).detach().cpu().numpy()).transpose(2, 3, 1, 0)
            target_vector = target_vector[:, :, :, 0]
            target_vector = cv2.resize(target_vector, (1280, 720))
            target_vector = cv2.circle(
                np.uint8(target_vector*255), region, 5, color, thickness=-1)
            target_vector = np.float32(target_vector)/255
            target_vector = target_vector.transpose(2, 0, 1)
            target_vector = target_vector[np.newaxis, :, :, :]

            target_video.append(target_vector)

            # print(frame_video[-1].shape,
            #       target_video[-1].shape, full_video[-1].shape)

        else:
            frame_video.append(un_frame.unsqueeze(0).detach().cpu().numpy())
            mask_video.append(mask.detach().cpu().numpy())
            traj_mask_video.append(
                skeletonize(
                    cv2.threshold(
                        traj_mask.detach().cpu().numpy().reshape(traj_mask.shape[2], traj_mask.shape[3]), args.traj_threshold, 1, cv2.THRESH_BINARY)[1]).reshape(
                            1, 1, traj_mask.shape[2], traj_mask.shape[3]))

            target_vector = np.copy(
                un_frame.unsqueeze(0).detach().cpu().numpy()).transpose(2, 3, 1, 0)
            target_vector = target_vector[:, :, :, 0]
            target_vector = cv2.resize(target_vector, (1280, 720))
            target_vector = np.float32(target_vector)/255
            target_vector = target_vector.transpose(2, 0, 1)
            target_vector = target_vector[np.newaxis, :, :, :]
            target_video.append(target_vector)

            # print(frame_video[-1].shape,
            #       target_video[-1].shape, full_video[-1].shape)
            print(f'================SKIPPING THIS TIME================')

        if frame_count > 500:
            target_number += 1
            frame_count = 0
            print(
                f'------------------INCREMENTING TARGET COUNT TO {target_number}------------------')

        print(
            f'probs = N/A with pred_found = {pred_found} and num_preds = {num_preds}')
        # if target_number > 3:
        #     pred_found = 1


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(
                x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def best_pixel(segmentation_map, threshold, confidence, method="weighted_average"):
    global frame_count
    global target_number
    global pred_found
    global args

    # cv2.imshow(f'seg_map', segmentation_map)
    # cv2.waitKey(10)
    if method == "weighted_average" or method == "mean_wa":
        segmentation_map[segmentation_map < threshold] = 0
        labeler = segmentation_map.copy()
        labeler[labeler >= threshold] = 1
        labels, num_labels = measure.label(labeler, return_num=True)
        count = list()
        for l in range(num_labels):
            if method == "mean_wa":
                count.append([l+1, np.mean(segmentation_map[labels == l+1])])
            else:
                count.append([l+1, np.sum(segmentation_map[labels == l+1])])
        count = np.array(count)
        if num_labels == 0:
            return -1
        count = count.reshape(num_labels, 2)
        largest_label = count[np.argmax(count[:, 1]), 0]
        print(
            f"================{count[np.argmax(count[:, 1]),1]}================")

        segmentation_map[labels != largest_label] = 0
        pos = (np.argmax(
            segmentation_map@np.arange(segmentation_map.shape[1])), np.argmax(np.arange(segmentation_map.shape[0])@segmentation_map))
        ret_count = count[np.argmax(count[:, 1]), 1]
    elif method == "max":
        pos = np.where(segmentation_map == np.amax(segmentation_map))
        pos = (pos[0][0], pos[1][0])

        labeler = segmentation_map.copy()
        labeler[labeler >= threshold] = 1
        labels, num_labels = measure.label(labeler, return_num=True)
        count = list()
        for l in range(num_labels):
            count.append([l+1, np.sum(segmentation_map[labels == l+1])])
        count = np.array(count)
        if num_labels == 0:
            return -1

        ret_count = np.sum(segmentation_map[labels == labels[pos]])

    final = (pos[1], pos[0])
    final = (final[0]*1280/segmentation_map.shape[1],
             final[1]*720/segmentation_map.shape[0])
    final = (int(final[0]), int(final[1]))
    # final = pos
    return (ret_count, final)
    # return pos


def best_pixel_traj(traj_mask_np):
    candidates = np.where(traj_mask_np > 0)
    candidates = np.vstack([candidates[0], candidates[1]]).T
    if candidates.size == 0:
        candidates = np.array(
            [traj_mask_np.shape[0]-1, traj_mask_np.shape[1]/2-1]).reshape(1, 2)
    dist_mat = spatial.distance_matrix(candidates, np.array(
        [traj_mask_np.shape[0]-1, traj_mask_np.shape[1]/2-1]).reshape(1, 2))
    pos = (*candidates[np.argmax(dist_mat)].tolist(),)
    final = (pos[1], pos[0])
    final = (final[0]*1280/traj_mask_np.shape[1],
             final[1]*720/traj_mask_np.shape[0])
    final = (int(final[0]), int(final[1]))
    return (-1, final)


def world_to_pixel(K, rgb_matrix, destination,  curr_position):

    point_3d = np.ones((4, destination.shape[1]))
    point_3d[0] = destination[0]
    point_3d[1] = destination[1]
    point_3d[2] = curr_position[2]

    # point_3d = np.array([destination[0], destination[1], curr_position[2], 1])
    # point_3d = np.round(point_3d, decimals=2)
    # print("3D world coordinate: ", point_3d)

    cam_coords = rgb_matrix @ point_3d
    # cam_coords = rgb_matrix @ point_3d[:, None]
    cam_coords = np.array([cam_coords[1], cam_coords[2]*-1, cam_coords[0]])
    cam_coords = cam_coords[:, cam_coords[2, :] > 0]
    points_2d = np.dot(K, cam_coords)

    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]]
    )
    points_2d = points_2d.reshape(3, -1)
    # points_2d = np.round(points_2d, decimals=2)
    return points_2d


def pixel_to_world(image, vehicle_matrix, vehicle_location, weak_agent, screen_pos, K, destination, set_destination=True):
    global command_given
    global target_number
    global new_destination

    agent_weak = weak_agent()

    # image.save_to_disk('_out/%06d.jpg' % image.frame)

    # image.convert(cc.Depth)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    im_array = array[:, :, ::-1]  # [:, :, 0]

    print(im_array.shape)

    depth_cam_matrix = image.transform.get_matrix()
    depth_cam_matrix_inv = image.transform.get_inverse_matrix()

    depth_cam_matrix = np.array(depth_cam_matrix)
    depth_cam_matrix_inv = np.array(depth_cam_matrix_inv)

    # vehicle_matrix = agent_weak._vehicle.get_transform().get_matrix()
    # vehicle_matrix_inv = agent_weak._vehicle.get_transform().get_inverse_matrix()

    vehicle_matrix = np.array(vehicle_matrix)

    # print("=========================")
    # print("Depth Camera Matrix:")
    # pprint(depth_cam_matrix)
    # print("Vehicle Matrix:")
    # pprint(vehicle_matrix)
    # print("=========================")

    # print("Pixel Coords: ", screen_pos)

    screen_pos = (min(max(0, screen_pos[0]), im_array.shape[1]), min(
        max(0, screen_pos[1]), im_array.shape[0]))

    R, G, B = im_array[screen_pos[1], screen_pos[0]]
    # print(im_array.shape, 'Screen pos max vals, order: 1,0')
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    depth = 1000 * normalized

    # print("Depth: ", depth)

    pos_2d = np.array([screen_pos[0], screen_pos[1], 1])
    # print("2D Pixel Coords Homogenous: ", pos_2d)

    # print("Camera Intrinsic Matrix:\n", K)

    pos_3d__ = np.linalg.inv(K) @ pos_2d[:, None] * depth
    pos_3d__ = pos_3d__.reshape(-1)
    # print("Camera Coordinates: ", pos_3d__)

    # Order Change
    pos_3d_ = np.array([pos_3d__[2], pos_3d__[0], pos_3d__[1]])
    # print("After Camera Coordinates: ", pos_3d_)

    pos_3d_ = np.array([pos_3d_[0], pos_3d_[1], pos_3d_[2], 1])

    pos_3d_ = depth_cam_matrix @ pos_3d_[:, None]
    pos_3d_ = pos_3d_.reshape(-1)
    # print("After Camera Matrix World Coordinates: ", pos_3d_)

    X = (screen_pos[0] - K[0, 2]) * (depth / K[0, 0])
    Y = (screen_pos[1] - K[1, 2]) * (depth / K[1, 1])
    Z = depth

    xyz_pos_ = np.array([X, Y, Z, 1])
    xyz_pos = vehicle_matrix @ depth_cam_matrix @ xyz_pos_[:, None]
    xyz_pos = xyz_pos.reshape(-1)

    new_destination = carla.Location(
        x=pos_3d_[0], y=pos_3d_[1], z=vehicle_location.z)

    if set_destination:
        agent_weak.set_destination(new_destination)

        print("=======================================")
        print(f"old destination : {destination}")
        print(f"new destination : {new_destination}")
        print(
            f"vehicle position: {agent_weak._vehicle.get_transform().location}")
        print(f"Est. position   : {xyz_pos}")
        print("=======================================")

    time.sleep(0.5)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    global saving
    global command_given
    global episode_number
    global agent
    global depth_camera
    global target_number
    global frame_count
    global weak_dc
    global weak_agent
    global K
    global destination
    global command
    global world
    global clock
    global frame_pending

    global network
    global corpus
    global img_transform
    global unnormalized_transform
    global mask_transform
    global traj_transform
    global phrase
    global phrase_mask
    global frame_mask
    global threshold
    global confidence
    global mode

    global pred_found
    global num_preds

    global frame_video
    global traj_mask_video
    global target_video
    global mask_video
    global full_video

    global depth_cam_queue
    global rgb_cam_queue

    global prev_loc

    depth_cam_queue = queue.Queue()
    rgb_cam_queue = queue.Queue()

    pygame.init()
    pygame.font.init()
    world = None

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        client.load_world(
            f'{args.map}', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        blueprint_lib = client.get_world().get_blueprint_library()
        camera_bp = blueprint_lib.filter("sensor.camera.depth")[0]
        # print(blueprint_lib.filter('vehicle.*.*'))

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        actor_list = []
        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        ######### spawn vehicles begin #########
        spawn_points = world.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        print('found %d spawn points.' % len(spawn_points))

        count = 30

        blueprints = world.world.get_blueprint_library().filter('vehicle.*')

        blueprints = [x for x in blueprints if int(
            x.get_attribute('number_of_wheels')) == 4]
        blueprints = [
            x for x in blueprints if not x.id.endswith('microlino')]
        blueprints = [
            x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [
            x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = [
            x for x in blueprints if not x.id.endswith('sprinter')]
        blueprints = [
            x for x in blueprints if not x.id.endswith('firetruck')]
        blueprints = [
            x for x in blueprints if not x.id.endswith('ambulance')]

        spawn_points = world.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if count < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif count > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, count, number_of_spawn_points)
            count = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= count:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, transform).then(
                SetAutopilot(FutureActor, True)))

        for response in client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                actor_list.append(response.actor_id)

        ######### spawn vehicles end #########

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        walkers_list = []
        all_id = []
        blueprintsWalkers = get_actor_blueprints(
            world.world, 'walker.pedestrian.*', '2')

        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        # how many pedestrians will walk through the road
        percentagePedestriansCrossing = 0.0
        # 1. take all the random locations to spawn
        spawn_points = []
        number_of_walkers = 70
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute(
                        'speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute(
                        'speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp,
                         carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if not args.sync:
            world.world.wait_for_tick()
        else:
            world.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(
                world.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        ############### SPAWN WALKERS END ###############

        if args.agent == "Basic":
            agent = BasicAgent(world.player)
        else:
            agent = BehaviorAgent(world.player, behavior=args.behavior)

        # Ignore Rules
        # agent.ignore_traffic_lights(True)
        # agent.ignore_stop_signs(True)
        # agent.follow_speed_limits(True)
        # agent.set_target_speed(10)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        # agent.set_destination(destination)

        agent.target_destination = None

        print(f"Destination is {destination}!")

        clock = pygame.time.Clock()

        camera_manager = world.camera_manager

        # rgb_matrix = world.camera_manager.sensor.get_transform().get_matrix()
        # rgb_matrix = np.array(rgb_matrix)
        # rgb_matrix = np.round(rgb_matrix, decimals=2)

        # Getting the Depth Sensor
        depth_sensor_info = world.camera_manager.sensors[2]
        depth_camera = world.player.get_world().spawn_actor(
            depth_sensor_info[-1],
            camera_manager._camera_transforms[0][0],
            attach_to=world.player,
            attachment_type=camera_manager._camera_transforms[0][1])

        image_w = args.width
        image_h = args.height
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

        # Calculating the Calibration Matrix
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        print("Intinsic Matrix:\n", K)

        temp_dir = "./_out"
        os.makedirs(temp_dir, exist_ok=True)

        handled = False

        # for file_ in os.listdir(temp_dir):
        #     shutil.rmtree(os.path.join(temp_dir, file_))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpu = torch.cuda.device_count()
        print(f"Using {device} with {num_gpu} GPUS!")

        solver = Solver(args, inference=True, force_parallel=True)

        glove_path = args.glove_path
        checkpoint_path = args.checkpoint

        corpus = Corpus(glove_path)
        if args.img_backbone == 'conv3d_baseline':
            feature_dim = 7
        else:
            feature_dim = 14

        frame_mask = torch.ones(
            1, feature_dim * feature_dim, dtype=torch.int64).cuda(non_blocking=True)

        threshold = args.threshold
        confidence = args.confidence

        mode = solver.mode
        network = solver.network

        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint["state_dict"])

        network.eval()

        img_transform = solver.val_transform
        unnormalized_transform = solver.unnormalized_transform

        mask_transform = solver.mask_transform

        traj_transform = solver.traj_transform

        command_given = False
        # currently saving, need to start next episode, delete current episode
        saving = [True, True, False]
        if len(os.listdir(temp_dir)) == 0:
            episode_number = -1
        else:
            episode_number = max([int(x) for x in os.listdir(temp_dir)])
        if args.spawn != -1:
            episode_number = args.spawn-1
        target_number = 0
        frame_count = 0
        stationary_frames = 0
        frame_pending = 0
        pred_found = 0
        frames_from_done = 0
        checked = False
        num_preds = 0
        curr_times = 0
        full_video = []
        prev_loc = None

        time_since_stopped = 0
        time_since_running = 0

        weak_dc = weakref.ref(depth_camera)
        weak_agent = weakref.ref(agent)

        depth_camera.listen(depth_cam_queue.put)
        done = False
        new_start = True
        if args.infer_dataset == 'val5':
            commands = ['Go straight and stop before the traffic light.',
                        'go straight and take a right',
                        'Shift to the left lane and stop at the intersection.',
                        'Stop after crossing the stop signboard',
                        'Park on the roadside after you have crossed the bus stop',
                        'Take the next right turn.',
                        'Stop in front of the garbage dumpsters.',
                        'change to left lane',
                        'Go straight and stop a few metres ahead of the bus stop',
                        'Stop near the signboard',
                        'stop by the lamp post',
                        'After crossing the blue car come to a stop',
                        'Drive straight till the intersection and then turn left',
                        'Drive to the corner and go right',
                        'Turn right from the intersection.',
                        'Stop near the red shelters',
                        'Drive straight till you see a red coloured board',
                        'take right and stop near the speed limit sign',
                        'take a right turn and stop across the bench on left',
                        'Stop before the next right turn.',
                        'Park just ahead of the speed signboard',
                        'Turn right and stop near the bus stop.',
                        'Wait at the traffic light then go straight.',
                        'Turn right from the intersection',
                        'park immediately after the signboard',
                        'Wait for the signal to turn green and continue straight',
                        'park beside the red volkswagen',
                        'Go straight and stop next to the grey gate.',
                        'stop at the bus stand in front',
                        'Drive towards the bus stop and park after crossing it',
                        'Park near the hut visible on the right',
                        'go straight and take a left',
                        'Take a left from the interection.',
                        'stop beside the black suv',
                        'take a right at the traffic lights and then take a left',
                        'take a right at the intersection',
                        'Stop as soon as you encounter a white car',
                        'Turn right from the crossroads ahead.',
                        'Drive straight then turn right.',
                        'Turn left and stop next to the bus stop.',
                        'Turn left from the intersection',
                        'Stop near the scooty',
                        'Turn left from the next crossroads.',
                        'Park near the red signboard',
                        'Drive to the intersection and turn left',
                        'Drive straight from the next crossroads.',
                        'Turn right and park near the sidewalk.',
                        'Change lanes and stop at the junction.',
                        'Turn right from the crossroads.',
                        'Take a left from the intersection ahead.']
            sub_commands = [
                ['park near the bus stand'],
                ['stop near the tallest building'],
                ['Take a right from the intersection'],
                ['Go right from the corner'],
                ['Drive towards the bus stop'],
                ['Take a left from the intersection.'],
                ['take a right', 'stop near the pedestrian'],
                ['stop beside the black suv'],
                ['Wait for the signal to turn green', 'then go straight'],
                ['Go straight from the intersection',
                    'stop next to the bus stop.'],
                ['Go straight', 'park behind the first car you see'],
                ['stop by the lamp pole'],
                ['stop across the house with stairs'],
                ['stop in front of the maroon car in rightmost lane'],
                ['Go right from the corner'],
                ['park behind the brown car'],
                ['take a right', 'stop near the man in blue'],
                ['take a right at the intersection'],
                ['take a right at the traffic lights', 'then take a left'],
                ['Take the road on the left'],
                ['wait for traffic light', ' take left'],
                ['Stop as soon as you encounter a white car'],
                ['Stop near the blue dustbin which you see in front'],
                ['Wait for the green signal', 'take a left from the intersection.'],
                ['Stop in front of the white car']
                # ['take a right', 'go left from the traffic lights',
                #     'take a right turn', 'stop near the bus stop'],
            ]
        if args.infer_dataset == 'val':
            commands = ['Turn left and stop next to the bus stop.',
                        'Take a left from the interection.',
                        'Drive straight till you see a red coloured board',
                        'Park just ahead of the speed signboard',
                        'Take the next right turn.',
                        'Drive straight from the next crossroads.',
                        'change to left lane',
                        'Stop near the scooty',
                        'take right and stop near the speed limit sign',
                        'Turn right and stop near the bus stop.',
                        'Shift to the left lane and stop at the intersection.',
                        'Stop after crossing the stop signboard',
                        'Park near the red signboard',
                        'Wait at the traffic light then go straight.',
                        'Drive straight till the intersection and then turn left',
                        'go straight and take a left',
                        'Stop as soon as you encounter a white car',
                        'take a right at the intersection',
                        'Stop near the red shelters',
                        'Turn left from the intersection',
                        'Drive straight then turn right',
                        'Park near the hut visible on the right',
                        'Go straight and stop before the traffic light.',
                        'stop at the bus stand in front',
                        'Turn right from the intersection.']
            sub_commands = commands
        if args.infer_dataset == 'test':
            commands = ['Stop before the next right turn.',
                        'After crossing the blue car come to a stop',
                        'stop beside the black suv',
                        'Stop near the signboard',
                        'Take a left at the traffic light',
                        'Turn right from the intersection',
                        'Take a left from the intersection ahead',
                        'Drive to the corner and go right',
                        'Drive to the intersection and turn left',
                        'Go straight and stop a few metres ahead of the bus stop',
                        'Stop in front of the garbage dumpsters.',
                        'Go straight and stop next to the grey arch.',
                        'Change lanes and stop at the junction.',
                        'Park on the roadside after you have crossed the bus stop',
                        'Wait for the signal to turn green and continue straight',
                        'Drive towards the bus stop and park after crossing it',
                        'take a right turn and stop across the bench on left',
                        'Turn right from the crossroads ahead',
                        'Turn right and park near the sidewalk.',
                        'go straight and take a right',
                        'park beside the red volkswagen',
                        'Turn left from the next crossroads',
                        'Turn right from the crossroads.',
                        'stop by the lamp post',
                        'park immediately after the signboard',
                        'Turn right and stop beside the traffic signal',
                        'Turn left and stop near the traffic signal',
                        'Turn left and stop near the intersection',
                        'Turn left and come to a halt at the intersection',
                        'Turn right and stop before the traffic signal',
                        'Turn right and stop just before the traffic signal',
                        'Turn right and stop near the bus stop',
                        'Turn left and stop beside the traffic signal',
                        'Turn left and stop near the traffic signal']
            sub_commands = commands
        if args.sub_command:
            assert args.command == True
            assert args.spawn != -1
        if args.sub_command:
            times_check = args.num_preds
            args.num_preds = len(sub_commands[args.spawn])
        while not done:
            clock.tick()
            if args.sync:
                world.world.tick()
            else:
                world.world.wait_for_tick()
            if controller.parse_events():
                return

            curr_position = agent._vehicle.get_transform().location

            if saving[2]:
                try:
                    shutil.rmtree(f'_out/{episode_number}')
                    print(f'Deleted _out/{episode_number}')
                except:
                    print(f'Unable to delete _out/{episode_number}')
                saving[2] = False

            if (not args.command and pygame.mouse.get_pressed()[0] and not handled) or (new_start and args.command):
                # if not command_given:
                if saving[0]:
                    if saving[1]:
                        saving[1] = False
                        target_number = 0
                        pred_found = 0
                        frame_count = 0
                        stationary_frames = 0
                        frames_from_done = 0
                        num_preds = 0
                        full_video = []
                        episode_number += 1
                        try:
                            if os.path.exists(f'_out/{episode_number}'):
                                shutil.rmtree(f'_out/{episode_number}')
                            else:
                                print('Failed to delete')
                        except:
                            print('Failed to delete')
                        os.makedirs(f'_out/{episode_number}', exist_ok=True)
                        if not args.command:
                            command = input('Enter Command: ')
                        elif not args.sub_command:
                            command = commands[args.spawn]
                        else:
                            command = sub_commands[args.spawn][num_preds]
                        print(command)
                        # command = 'a'
                        with open(f'_out/{episode_number}/command.txt', 'w') as f:
                            f.write(command)
                        np.save(
                            f'_out/{episode_number}/camera_intrinsic.npy', K)
                        command = re.sub(r"[^\w\s]", "", command)

                        if 'clip_' not in args.img_backbone:
                            phrase, phrase_mask = corpus.tokenize(command)
                            phrase = phrase.unsqueeze(0).cuda()
                            phrase_mask = phrase_mask.unsqueeze(0).cuda()
                        else:
                            phrase = clip.tokenize(command).cuda()
                            phrase_mask = phrase.detach().clone()
                            phrase_mask[phrase_mask != 0] = 1
                            phrase_mask = phrase_mask.detach().clone().cuda()

                        command_given = True
                        new_start = False

                        prev_loc = None
                        prev_prev_loc = None

                        # print('Processing FIRST frame')
                        # measurements, sensor_data = client.read_data()
                        # print(type(sensor_data))
                        # process_network(measurements, sensor_data,
                        #                 agent._vehicle.get_transform().get_matrix())

                    # episode_number = -episode_number

                # screen_pos = pygame.mouse.get_pos()

                # Listening to Depth Sensor Data
                # weak_dc = weakref.ref(depth_camera)
                # weak_agent = weakref.ref(agent)

                # depth_camera.listen(lambda image: pixel_to_world(
                #     image, weak_dc, weak_agent, screen_pos, K, destination))

                # else:
                #     print('In route')

            if command_given == False and args.sub_command and not new_start:
                command = sub_commands[args.spawn][num_preds]
                print(command)
                # command = 'a'
                with open(f'_out/{episode_number}/command.txt', 'a+') as f:
                    f.write(command)
                command = re.sub(r"[^\w\s]", "", command)

                phrase, phrase_mask = corpus.tokenize(command)
                phrase = phrase.unsqueeze(0).cuda()
                phrase_mask = phrase_mask.unsqueeze(0).cuda()

                command_given = True
                print('Assigned')

            handled = pygame.mouse.get_pressed()[0]

            if not depth_cam_queue.empty() and not rgb_cam_queue.empty():
                depth_cam_data = depth_cam_queue.get()
                rgb_cam_data = rgb_cam_queue.get()
                vehicle_transform = agent._vehicle.get_transform()
                vehicle_matrix = vehicle_transform.get_matrix()
                vehicle_location = vehicle_transform.location

                if command_given and not (args.sub_command and curr_times >= times_check):
                    if not pred_found and num_preds < args.num_preds:
                        print_network_stats = 1
                    start = time.time()
                    if args.sampling_type == 'linear':
                        sampling_multiplier = (
                            2*curr_times+1 if args.sub_command else 2*num_preds+1)
                    elif args.sampling_type == 'constant':
                        sampling_multiplier = 1
                    process_network(rgb_cam_data, depth_cam_data, vehicle_matrix,
                                    vehicle_location, args.sampling*sampling_multiplier)
                    end = time.time()

                    if prev_loc is not None and ((prev_loc.x - vehicle_location.x)**2 + (prev_loc.y - vehicle_location.y)**2)**0.5 < args.min_distance:
                        pred_found = 0
                        stationary_frames += 1
                        time_since_stopped += 1
                        time_since_running = 0
                    else:
                        time_since_running += 1
                        time_since_stopped = 0
                        prev_loc = vehicle_location
                    if frame_count % args.sampling == 0 and print_network_stats:
                        print(
                            f'Network took {end-start}, pred_found = {pred_found}, curr_times = {curr_times}')
                        # print(
                        #     f'++++++++++++++++++++++++++++++++frame_count:{frame_count} out of {1500+stationary_frames}++++++++++++++++++++++++++++++++')
                        if args.sub_command:
                            if pred_found >= 0:
                                curr_times += pred_found
                            else:
                                curr_times = 1
                            # if curr_times >= times_check:
                            #     num_preds += 1
                        else:
                            if pred_found >= 0:
                                num_preds += pred_found
                            else:
                                num_preds = 1
                    if pred_found:
                        print(
                            f'-------------Num Preds: {num_preds}-------------')

                    if num_preds > 0:
                        frames_from_done += 1
                    frame_count += 1
                prev_prev_loc = prev_loc
                pred_found = 0
                if not args.sub_command:
                    pred_found = 0

            # if target_number > 5:
            #     pred_found = 1
            #     target_number = 0
            #     frame_count = 0
            if (agent.done() and prev_loc == prev_prev_loc and command_given) or frame_count > 1000+stationary_frames or frame_count > 2000:
                pred_found = 0
                if args.sub_command and command_given:
                    command_given = False
                    curr_times = 0
                    num_preds += 1
                    print('Reached')
                if frame_count > 1500+stationary_frames or frame_count > 3000:
                    num_preds = args.num_preds
                if time_since_stopped > 120 and agent.done():
                    num_preds = args.num_preds
                if num_preds >= args.num_preds:
                    command_given = False
                    saving = [True, True, False]
                    print('Episode Done')
                    pred_found = 0
                    num_preds = 0

                    for full_video_index in range(len(full_video)):
                        pass

                    frame_video = np.concatenate(frame_video, axis=0)
                    full_video = np.concatenate(full_video, axis=0)
                    target_video = np.concatenate(target_video, axis=0)
                    mask_video = np.concatenate(mask_video, axis=0)
                    traj_mask_video = np.concatenate(traj_mask_video, axis=0)

                    # import pdb; pdb.set_trace()
                    mask_video_overlay = np.copy(frame_video)
                    mask_video_overlay[:,
                                       0] += (mask_video[:, 0]/mask_video.max())
                    if mask_video.shape[1] == 2:
                        mask_video_overlay[:,
                                           1] += (mask_video[:, 1]/mask_video.max())
                    mask_video_overlay = np.clip(
                        mask_video_overlay, a_min=0., a_max=1.)

                    traj_mask_video_overlay = np.copy(frame_video)
                    traj_mask_video_overlay[:,
                                            0] += (traj_mask_video[:, 0]/traj_mask_video.max())
                    traj_mask_video_overlay = np.clip(
                        traj_mask_video_overlay, a_min=0., a_max=1.)

                    frame_video = np.uint8(frame_video * 255)
                    full_video = np.uint8(full_video * 255)
                    target_video = np.uint8(target_video * 255)
                    mask_video = np.uint8(mask_video_overlay * 255)
                    traj_mask_video = np.uint8(traj_mask_video_overlay * 255)

                    # print(full_video.shape, frame_video.shape)

                    wandb.log(
                        {
                            "full_video": wandb.Video(full_video, fps=10, caption=command, format="mp4"),
                            "video": wandb.Video(frame_video, fps=1, caption=command, format="mp4"),
                            "target video": wandb.Video(target_video, fps=1, caption=command, format="mp4"),
                            "pred_mask": wandb.Video(mask_video, fps=1, caption=command, format="mp4"),
                            "traj_mask": wandb.Video(traj_mask_video, fps=1, caption=command, format="mp4"),
                        }
                    )
                    frame_video = []
                    mask_video = []
                    target_video = []
                    traj_mask_video = []
                    full_video = []
                    if args.command:
                        done = True

            if agent.target_destination:
                destination = agent.target_destination
                curr_position = vehicle_location
                distance = np.sqrt((destination.x - curr_position.x)**2 +
                                   (destination.y - curr_position.y)**2)

                points_2d = []

                # x_offsets = np.linspace(-0.5, 0.5, num=25)
                # y_offsets = np.linspace(-0.5, 0.5, num=25)
                x_offsets = np.linspace(-0.0, 0, num=1)
                y_offsets = np.linspace(-0, 0, num=1)
                X, Y = np.meshgrid(x_offsets, y_offsets)

                mesh = np.dstack([X, Y])

                mesh = mesh.reshape(-1, 2)

                mesh = np.hstack([mesh, np.zeros((mesh.shape[0], 1))]).T
                dest = np.array([destination.x, destination.y, destination.z])

                rgb_camera = world.camera_manager.sensor
                rgb_matrix = rgb_cam_data.transform.get_inverse_matrix()[
                    :3]

                pos = np.array(
                    [curr_position.x, curr_position.y, curr_position.z])

                annotations = world_to_pixel(
                    K, rgb_matrix, dest.reshape(3, 1)+mesh, pos).T

                for i in range(annotations.shape[0]):
                    points_2d.append(annotations[i])
                    # pprint(annotations[i])

                world.tick(clock)
                world.render(display)

                for point in annotations:
                    pygame.draw.circle(display, (0, 255, 0),
                                       (point[0], point[1]), 10)
                pygame.display.flip()
            else:
                world.tick(clock)
                world.render(display)
                pygame.display.flip()
            # world.tick(clock)
            # world.render(display)
            # pygame.display.flip()

            # if agent.done():
            #     print("reached destination: ", vehicle_pos)
            #     if args.loop:
            #         agent.set_destination(random.choice(spawn_points).location)
            #         world.hud.notification(
            #             "The target has been reached, searching for another target", seconds=4.0)
            #         print("The target has been reached, searching for another target")
            #     else:
            #         print("The target has been reached, stopping the simulation")
            #         break

            if not agent.done():
                control = agent.run_step()
                control.manual_gear_shift = False
                if agent.target_destination:
                    world.player.apply_control(control)
                    # pass

    finally:
        cv2.destroyAllWindows()
        print('\ndestroying %d actors' % len(actor_list))
        client.apply_batch_sync([carla.command.DestroyActor(x)
                                for x in actor_list])

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    global args
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-m', '--map',
        metavar='M',
        default='Town10HD',
        choices=[f'Town{x:02d}' for x in [1, 2, 3, 4, 5, 6, 7]]+['Town10HD'],
        type=str,
        help='World map (default: Town10)')
    argparser.add_argument(
        '--spawn',
        default=-1,
        type=int,
        help='Spawn Point (default: Random)')
    argparser.add_argument(
        '--infer_dataset',
        required=True,
        type=str,
        choices=['val', 'test', 'val5'],
        help='Infer Dataset')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.tt',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    argparser.add_argument(
        "--glove_path",
        default="/ssd_scratch/cvit/kanishk/glove",
        type=str,
        help="dataset name",
    )

    argparser.add_argument(
        "--model",
        default='baseline',
        choices=[
            'baseline'
        ],
        type=str,
    )

    argparser.add_argument(
        "--command",
        action="store_true"
    )

    argparser.add_argument(
        "--sub_command",
        action="store_true"
    )

    argparser.add_argument(
        "--attn_type",
        default='dot_product',
        choices=[
            'dot_product',
            'scaled_dot_product',
            'multi_head',
            'rel_multi_head',
            'custom_attn'
        ],
        type=str,
    )

    argparser.add_argument(
        "--imtext_matching",
        default='cross_attention',
        choices=[
            'cross_attention',
            'concat',
            'avg_concat',
        ],
        type=str,
    )

    argparser.add_argument(
        "--img_backbone",
        default="vit_tiny_patch16_224",
        choices=[
            "vit_tiny_patch16_224",
            "vit_small_patch16_224",
            "vit_tiny_patch16_384",
            "vit_small_patch16_384",
            "dino_resnet50",
            "timesformer",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
            "deeplabv3_mobilenet_v3_large",
            "conv3d_baseline",
            "clip_ViT-B/16",
            "clip_ViT-B/32",
            "clip_ViT-L/16",
            "clip_ViT-L/32",
            "clip_ViT-L/14@336px",
        ],
        type=str,
    )

    argparser.add_argument(
        "--target",
        default="mask",
        choices=[
            "mask",
            'mask_dual',
            "trajectory",
            "network",
            "network2",
            "distance",
        ],
        type=str,
    )

    argparser.add_argument(
        "--stop_criteria",
        default="confidence",
        choices=[
            "confidence",
            "distance",
                "consistent",
        ],
        type=str,
    )
    argparser.add_argument(
        "--sampling_type",
        default="linear",
        choices=[
            "linear",
            "constant",
        ],
        type=str,
    )

    argparser.add_argument(
        '--distance',
        default=0.5,
        type=float,
    )

    argparser.add_argument("--image_dim", type=int,
                           default=448, help="Image Dimension")
    argparser.add_argument("--mask_dim", type=int,
                           default=448, help="Mask Dimension")
    argparser.add_argument("--traj_dim", type=int,
                           default=448, help="Traj Dimension")
    argparser.add_argument("--hidden_dim", type=int,
                           default=256, help="Hidden Dimension")
    argparser.add_argument("--num_frames", type=int,
                           default=16, help="Frames of Video")
    argparser.add_argument("--traj_frames", type=int,
                           default=16, help="Next Frames of Trajectory")
    argparser.add_argument("--traj_size", type=int,
                           default=25, help="Trajectory Size")
    argparser.add_argument("--one_in_n", type=int,
                           default=20, help="Image Dimension")
    argparser.add_argument("--patch_size", type=int,
                           default=16, help="Patch Size of Video Frame for ViT")

    argparser.add_argument("--checkpoint", type=str)

    argparser.add_argument("--traj_threshold", type=float,
                           default=0.4, help="mask threshold")

    argparser.add_argument("--threshold", type=float,
                           default=0.005, help="mask threshold")

    argparser.add_argument("--confidence", type=float,
                           default=100, help="mask confidence")

    argparser.add_argument("--min_confidence", type=float,
                           default=0.3, help="mask confidence")
    argparser.add_argument("--min_distance", type=float,
                           default=5e-1, help="mask confidence")
    argparser.add_argument("--considered_distance", type=float,
                           default=20, help="mask confidence")

    argparser.add_argument("--confidence_probs", type=float,
                           default=0.5, help="mask confidence")

    argparser.add_argument("--sampling", type=float,
                           default=20, help="mask confidence")

    argparser.add_argument("--num_preds", type=int,
                           default=3, help="mask confidence")

    argparser.add_argument("--save", default=False, action="store_true")

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    global command_given
    global saving
    global episode_number
    global agent
    global depth_camera
    global target_number
    global pred_found
    global frame_count
    global network
    global weak_dc
    global weak_agent
    global frame_pending

    command_given = False
    saving = [True, True, False]
    episode_number = 0

    main()
