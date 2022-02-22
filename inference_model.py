#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

# CarlaUE4.exe -windowed -carla-server -quality-level=Low

from __future__ import print_function
import shutil

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


import torch
import timm
from timesformer.models.vit import VisionTransformer
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter

from models.model import *
from dataloader.word_utils import Corpus


import cv2
from skimage import measure
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
            spawn_point = random.choice(
                spawn_points) if spawn_points else carla.Transform()
            # Fix Spawning Point
            # spawn_point = spawn_points[0] if spawn_points else carla.Transform(
            # )
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
        global target_video
        global target_number
        global frame_count
        global command
        if key == K_d:
            saving[1] = True
            frame_video = np.concatenate(frame_video, axis=0)
            mask_video = np.concatenate(mask_video, axis=0)
            target_video = np.concatenate(target_video, axis=0)

            # import pdb; pdb.set_trace()
            mask_video_overlay = np.copy(frame_video)
            mask_video_overlay[:, 0] += (mask_video[:, 0]/mask_video.max())
            mask_video_overlay = np.clip(
                mask_video_overlay, a_min=0., a_max=1.)

            frame_video = np.uint8(frame_video * 255)
            target_video = np.uint8(target_video * 255)
            mask_video = np.uint8(mask_video_overlay * 255)
            print(frame_video.shape, mask_video.shape)

            wandb.log(
                {
                    "video": wandb.Video(frame_video, fps=1, caption=command, format="mp4"),
                    "target video": wandb.Video(target_video, fps=1, caption=command, format="mp4"),
                    "pred_mask": wandb.Video(mask_video, fps=1, caption=command, format="mp4"),
                }
            )
            frame_video = []
            mask_video = []
            target_number = 0
            frame_count = 0

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
        global phrase
        global phrase_mask
        global frame_mask
        global threshold
        global confidence

        global frame_video
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


def process_network(image, depth_cam_data, vehicle_matrix, vehicle_location):
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
    global phrase
    global phrase_mask
    global frame_mask
    global threshold
    global confidence

    global frame_video
    global mask_video
    global target_video

    global frame_pending

    img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    img = np.reshape(
        img, (image.height, image.width, 4))  # RGBA format
    img = img[:, :, :]  # BGR

    if frame_count < 20 and target_number == 0:
        frame_video = []
        mask_video = []
        target_video = []

    if frame_count % 20 == 0 and target_number <= 3:
        im = Image.fromarray(img[:, :, :3][:, :, ::-1])

        frame = img_transform(im).cuda(
            non_blocking=True).unsqueeze(0)

        mask = network(frame, phrase, frame_mask, phrase_mask)

        mask_np = mask.detach().cpu().numpy().transpose(2, 3, 1, 0)
        mask_np = mask_np.reshape(mask_np.shape[0], mask_np.shape[1])
        print(mask_np.shape, mask_np.max(), mask_np.min())
        # mask_np = cv2.resize(mask_np, (1280, 720))
        pixel_out = best_pixel(mask_np, threshold, confidence)

        if pixel_out != -1:

            probs, region = pixel_out

            region = (region[0]*1280/mask_np.shape[1],
                      region[1]*720/mask_np.shape[1])

            region = (int(region[0]), int(region[1]))

            if probs > confidence:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            pixel_to_world(depth_cam_data, vehicle_matrix, vehicle_location, weak_agent,
                           region, K, destination)

            frame_video.append(frame.detach().cpu().numpy())
            mask_video.append(mask.detach().cpu().numpy())

            target_vector = np.copy(
                frame.detach().cpu().numpy()).transpose(2, 3, 1, 0)
            target_vector = target_vector[:, :, :, 0]
            target_vector = cv2.resize(target_vector, (1280, 720))
            target_vector = cv2.circle(
                np.uint8(target_vector*255), region, 5, color, thickness=-1)
            target_vector = np.float32(target_vector)/255
            target_vector = target_vector.transpose(2, 0, 1)
            target_vector = target_vector[np.newaxis, :, :, :]

            target_video.append(target_vector)

            print(frame_video[-1].shape, target_video[-1].shape)

        else:
            frame_video.append(frame.detach().cpu().numpy())
            mask_video.append(mask.detach().cpu().numpy())

            target_vector = np.copy(
                frame.detach().cpu().numpy()).transpose(2, 3, 1, 0)
            target_vector = target_vector[:, :, :, 0]
            target_vector = cv2.resize(target_vector, (1280, 720))
            target_vector = np.float32(target_vector)/255
            target_vector = target_vector.transpose(2, 0, 1)
            target_vector = target_vector[np.newaxis, :, :, :]
            target_video.append(target_vector)

            print(frame_video[-1].shape, target_video[-1].shape)
            print(f'================SKIPPING THIS TIME================')

    if frame_count > 500:
        target_number += 1
        frame_count = 0
        print(
            f'------------------INCREMENTING TARGET COUNT TO {target_number}------------------')

    if target_number > 3:
        frame_video = np.concatenate(frame_video, axis=0)
        target_video = np.concatenate(target_video, axis=0)
        mask_video = np.concatenate(mask_video, axis=0)

        # import pdb; pdb.set_trace()
        mask_video_overlay = np.copy(frame_video)
        mask_video_overlay[:, 0] += (mask_video[:, 0]/mask_video.max())
        mask_video_overlay = np.clip(
            mask_video_overlay, a_min=0., a_max=1.)

        frame_video = np.uint8(frame_video * 255)
        target_video = np.uint8(target_video * 255)
        mask_video = np.uint8(mask_video_overlay * 255)

        wandb.log(
            {
                "video": wandb.Video(frame_video, fps=1, caption=command, format="mp4"),
                "target video": wandb.Video(target_video, fps=1, caption=command, format="mp4"),
                "pred_mask": wandb.Video(mask_video, fps=1, caption=command, format="mp4"),
            }
        )
        frame_video = []
        mask_video = []
        target_video = []

    frame_count += 1


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

    # cv2.imshow(f'seg_map', segmentation_map)
    # cv2.waitKey(10)
    if method == "weighted_average":
        segmentation_map[segmentation_map < threshold] = 0
        labeler = segmentation_map.copy()
        labeler[labeler >= threshold] = 1
        labels, num_labels = measure.label(labeler, return_num=True)
        count = list()
        for l in range(num_labels):
            count.append([l+1, np.sum(segmentation_map[labels == l+1])])
        count = np.array(count)
        if num_labels == 0:
            return -1
        count = count.reshape(num_labels, 2)
        largest_label = count[np.argmax(count[:, 1]), 0]
        print(
            f"================{count[np.argmax(count[:, 1]),1]}================")
        if count[np.argmax(count[:, 1]), 1] > confidence:
            frame_count = 0 if target_number < 3 else frame_count
            target_number = 3
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
        if ret_count > confidence:
            frame_count = 0 if target_number < 3 else frame_count
            target_number = 3

    final = (pos[1], pos[0])
    # final = pos
    print((final[0]*1280/segmentation_map.shape[1], final[1]
          * 720/segmentation_map.shape[1]), "------>")
    print((final[1]*1280/segmentation_map.shape[1], final[0]
          * 720/segmentation_map.shape[1]), '------>')
    return (ret_count, final)
    # return pos


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

    print("=========================")
    print("Depth Camera Matrix:")
    pprint(depth_cam_matrix)
    print("Vehicle Matrix:")
    pprint(vehicle_matrix)
    print("=========================")

    print("Pixel Coords: ", screen_pos)

    R, G, B = im_array[screen_pos[1], screen_pos[0]]
    print(im_array.shape, 'Screen pos max vals, order: 1,0')
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    depth = 1000 * normalized

    print("Depth: ", depth)

    pos_2d = np.array([screen_pos[0], screen_pos[1], 1])
    print("2D Pixel Coords Homogenous: ", pos_2d)

    print("Camera Intrinsic Matrix:\n", K)

    pos_3d__ = np.linalg.inv(K) @ pos_2d[:, None] * depth
    pos_3d__ = pos_3d__.reshape(-1)
    print("Camera Coordinates: ", pos_3d__)

    # Order Change
    pos_3d_ = np.array([pos_3d__[2], pos_3d__[0], pos_3d__[1]])
    print("After Camera Coordinates: ", pos_3d_)

    pos_3d_ = np.array([pos_3d_[0], pos_3d_[1], pos_3d_[2], 1])

    pos_3d_ = depth_cam_matrix @ pos_3d_[:, None]
    pos_3d_ = pos_3d_.reshape(-1)
    print("After Camera Matrix World Coordinates: ", pos_3d_)

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

    # curr_position = agent._vehicle.get_transform().location

    # pos = np.array(
    #     [curr_position.x, curr_position.y, curr_position.z])

    # w2px = world_to_pixel(K, depth_cam_matrix_inv, np.array(
    #     [agent_weak.target_destination.x, agent_weak.target_destination.y, agent_weak.target_destination.z]).reshape(3, 1), pos).T

    # print(w2px[:,:2]/w2px[:,2])
    # print(w2px)

    command_given = True
    print("=======================================")
    print(f"old destination : {destination}")
    print(f"new destination : {new_destination}")
    print(f"vehicle position: {agent_weak._vehicle.get_transform().location}")
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
    global phrase
    global phrase_mask
    global frame_mask
    global threshold
    global confidence

    global frame_video
    global target_video
    global mask_video

    global depth_cam_queue
    global rgb_cam_queue

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

        experiment = wandb.init(project="Language Navigation", dir="/tmp")

        # val_path = os.path.join(args.data_root, 'val/')
        glove_path = args.glove_path
        checkpoint_path = args.checkpoint

        corpus = Corpus(glove_path)
        frame_mask = torch.ones(
            1, 14 * 14, dtype=torch.int64).cuda(non_blocking=True)

        threshold = args.threshold
        confidence = args.confidence

        return_layers = {"layer2": "layer2",
                         "layer3": "layer3", "layer4": "layer4"}

        mode = "image"
        if "vit_" in args.img_backbone:
            img_backbone = timm.create_model(
                args.img_backbone, pretrained=True)
            visual_encoder = nn.Sequential(*list(img_backbone.children())[:-1])
            network = SegmentationBaseline(
                visual_encoder,
                hidden_dim=args.hidden_dim,
                mask_dim=args.mask_dim,
                backbone=args.img_backbone,
            )
        elif "dino_resnet50" in args.img_backbone:
            img_backbone = torch.hub.load(
                "facebookresearch/dino:main", "dino_resnet50")
            visual_encoder = IntermediateLayerGetter(
                img_backbone, return_layers)
            network = IROSBaseline(
                visual_encoder, hidden_dim=args.hidden_dim, mask_dim=args.mask_dim
            )
        elif "timesformer" in args.img_backbone:
            mode = "video"
            spatial_dim = args.image_dim//args.patch_size
            visual_encoder = VisionTransformer(img_size=args.image_dim, patch_size=args.patch_size,
                                               embed_dim=args.hidden_dim, depth=2, num_heads=8, num_frames=args.num_frames)
            network = VideoSegmentationBaseline(
                visual_encoder, hidden_dim=args.hidden_dim, mask_dim=args.mask_dim, spatial_dim=spatial_dim, num_frames=args.num_frames,
            )
        elif "deeplabv3_" in args.img_backbone:
            img_backbone = torch.hub.load(
                "pytorch/vision:v0.10.0", args.img_backbone, pretrained=True
            )
            visual_encoder = nn.Sequential(
                *list(img_backbone._modules["backbone"].children())
            )
            network = SegmentationBaseline(
                visual_encoder,
                hidden_dim=args.hidden_dim,
                mask_dim=args.mask_dim,
                backbone=args.img_backbone,
            )
        wandb.watch(network, log="all")

        network = nn.DataParallel(network)
        if num_gpu > 1:
            print("Using DataParallel mode!")
        network.to(device)

        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint["state_dict"])

        network.eval()

        img_transform = transforms.Compose(
            [
                transforms.Resize((args.image_dim, args.image_dim)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                #                      0.229, 0.224, 0.225]),
            ]
        )

        mask_transform = transforms.Compose(
            [
                transforms.Resize((args.mask_dim, args.mask_dim)),
                transforms.ToTensor(),
            ]
        )

        frame_mask = torch.ones(
            1, 14 * 14, dtype=torch.int64).cuda(non_blocking=True)

        command_given = False
        # currently saving, need to start next episode, delete current episode
        saving = [True, True, False]
        if len(os.listdir(temp_dir)) == 0:
            episode_number = -1
        else:
            episode_number = max([int(x) for x in os.listdir(temp_dir)])
        target_number = 0
        frame_count = 0
        frame_pending = 0
        checked = False

        weak_dc = weakref.ref(depth_camera)
        weak_agent = weakref.ref(agent)

        depth_camera.listen(depth_cam_queue.put)

        while True:
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

            if pygame.mouse.get_pressed()[0] and not handled:
                # if not command_given:
                if saving[0]:
                    if saving[1]:
                        saving[1] = False
                        target_number = 0
                        frame_count = 0
                        episode_number += 1
                        os.makedirs(f'_out/{episode_number}', exist_ok=True)
                        command = input('Enter Command: ')
                        # command = 'a'
                        with open(f'_out/{episode_number}/command.txt', 'w') as f:
                            f.write(command)
                        np.save(
                            f'_out/{episode_number}/camera_intrinsic.npy', K)
                        command = re.sub(r"[^\w\s]", "", command)

                        phrase, phrase_mask = corpus.tokenize(command)
                        phrase = phrase.unsqueeze(0)
                        phrase_mask = phrase_mask.unsqueeze(0)

                        command_given = True

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

            handled = pygame.mouse.get_pressed()[0]

            if not depth_cam_queue.empty() and not rgb_cam_queue.empty():
                depth_cam_data = depth_cam_queue.get()
                rgb_cam_data = rgb_cam_queue.get()
                vehicle_transform = agent._vehicle.get_transform()
                vehicle_matrix = vehicle_transform.get_matrix()
                vehicle_location = vehicle_transform.location
                if command_given:
                    start = time.time()
                    process_network(rgb_cam_data, depth_cam_data, vehicle_matrix,
                                    vehicle_location)
                    end = time.time()
                    if frame_count % 20 == 1:
                        print(f'Network took {end-start}')

            if target_number > 3:
                saving = [True, True, False]
                command_given = False

                frame_video = np.concatenate(frame_video, axis=0)
                mask_video = np.concatenate(mask_video, axis=0)
                target_video = np.concatenate(target_video, axis=0)

                # import pdb; pdb.set_trace()
                mask_video_overlay = np.copy(frame_video)
                mask_video_overlay[:, 0] += (mask_video[:, 0]/mask_video.max())
                mask_video_overlay = np.clip(
                    mask_video_overlay, a_min=0., a_max=1.)

                frame_video = np.uint8(frame_video * 255)
                target_video = np.uint8(target_video * 255)
                mask_video = np.uint8(mask_video_overlay * 255)
                print(frame_video.shape, mask_video.shape)

                wandb.log(
                    {
                        "video": wandb.Video(frame_video, fps=1, caption=command, format="mp4"),
                        "target video": wandb.Video(target_video, fps=1, caption=command, format="mp4"),
                        "pred_mask": wandb.Video(mask_video, fps=1, caption=command, format="mp4"),
                    }
                )
                frame_video = []
                mask_video = []
                target_video = []
                target_number = 0
                frame_count = 0

            if agent.done() and command_given:
                if target_number > 3:
                    command_given = False
                    saving = [True, True, False]
                    print('Episode Done')
                else:
                    print('Done')
                    saving = [True, False, False]
                    target_number += 1

            if agent.target_destination:
                destination = agent.target_destination
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
                rgb_matrix = rgb_camera.get_transform().get_inverse_matrix()[
                    :3]

                curr_position = rgb_camera.get_transform().location

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

            if command_given:
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
            "deeplabv3_mobilenet_v3_large"
        ],
        type=str,
    )

    argparser.add_argument("--image_dim", type=int,
                           default=448, help="Image Dimension")
    argparser.add_argument("--mask_dim", type=int,
                           default=448, help="Mask Dimension")
    argparser.add_argument("--hidden_dim", type=int,
                           default=256, help="Hidden Dimension")
    argparser.add_argument("--num_frames", type=int,
                           default=16, help="Frames of Video")
    argparser.add_argument("--patch_size", type=int,
                           default=16, help="Patch Size of Video Frame for ViT")

    argparser.add_argument("--checkpoint", type=str)

    argparser.add_argument("--threshold", type=float,
                           default=0.4, help="mask threshold")

    argparser.add_argument("--confidence", type=float,
                           default=100, help="mask confidence")

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
    global frame_count
    global network
    global weak_dc
    global weak_agent
    global frame_pending

    command_given = False
    saving = [True, True, False]
    episode_number = 0

    main()
