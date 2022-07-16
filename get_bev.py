#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""

import glob
import os
import sys

import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import numpy as np


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


def get_map_loc(args):
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
    return corresponding_maps[args.spawn], other_spawns[args.spawn]


class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None


class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(
            sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.save_cnt = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(
                camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('range', '100')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute(
                'dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute(
                'dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute(
                'dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(
                lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar

        elif sensor_type == 'SemanticLiDAR':
            lidar_bp = self.world.get_blueprint_library().find(
                'sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '100')

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(
                lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_semanticlidar_image)

            return lidar

        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(
                radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)

            return radar

        else:
            return None

    def get_sensor(self):
        global call_exit
        return self.sensor

    def save_rgb_image(self, image):
        global call_exit
        self.save_cnt += 1
        t_start = self.timer.time()
        print(image.transform.location, image.transform.rotation)

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        # array = array[:, :, ::-1]

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')

        image_w = image.width
        image_h = image.height
        fov = camera_bp.get_attribute("fov").as_float()
        focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = image_w / 2.0
        K[1, 2] = image_h / 2.0

        if self.save_cnt == 3:
            cv2.imwrite(
                f'_bev/{args.infer_dataset}_{args.spawn}/image.png', array)
            np.save(f'_bev/{args.infer_dataset}_{args.spawn}/inverse_matrix.npy',
                    np.array(image.transform.get_inverse_matrix()))
            np.save(
                f'_bev/{args.infer_dataset}_{args.spawn}/camera_intrinsic.npy', K)
            call_exit = True

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

        if self.tics_processing > 50:
            pass
            # call_exit = True
        # call_exit = True

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_semanticlidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()


def run_simulation(args, spawn_loc, client):
    global call_exit
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('vehicle.*')[0]
        vehicle = world.spawn_actor(bp, random.choice(
            world.get_map().get_spawn_points()))

        vehicle_list.append(vehicle)
        vehicle.set_autopilot(False)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 1], window_size=[
                                         args.width, args.height])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position,
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=100), carla.Rotation(pitch=-90)),
        #               vehicle, {}, display_pos=[0, 0])
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=spawn_loc[0], y=spawn_loc[1], z=200), carla.Rotation(pitch=-90, roll=spawn_loc[3])),
                      None, {}, display_pos=[0, 0])
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+00)),
        #               vehicle, {}, display_pos=[0, 1])
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=+90)),
        #               vehicle, {}, display_pos=[0, 2])
        # SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)),
        #               vehicle, {}, display_pos=[1, 1])

        # SensorManager(world, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=2.4)),
        #               vehicle, {'channels': '64', 'range': '100',  'points_per_second': '250000', 'rotation_frequency': '20'}, display_pos=[1, 0])
        # SensorManager(world, display_manager, 'SemanticLiDAR', carla.Transform(carla.Location(x=0, z=2.4)),
        #               vehicle, {'channels': '64', 'range': '100', 'points_per_second': '100000', 'rotation_frequency': '20'}, display_pos=[1, 2])

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x)
                           for x in vehicle_list])

        world.apply_settings(original_settings)


def main():
    global args
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
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
        '--spawn',
        default=0,
        type=int,
        help='spawn episode number')
    argparser.add_argument(
        '--infer_dataset',
        default='val',
        choices=['val', 'test', 'val5', 'sample'],
        type=str,
        help='Dataset')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    map, spawn_loc = get_map_loc(args)

    os.makedirs('_bev', exist_ok=True)
    os.makedirs(f'_bev/{args.infer_dataset}_{args.spawn}', exist_ok=True)
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        client.load_world(
            f'{map}', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        run_simulation(args, spawn_loc, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    global call_exit
    call_exit = False
    main()
