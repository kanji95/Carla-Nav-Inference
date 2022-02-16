from email.policy import default
import numpy as np
import cv2
import argparse
import os
import sys
import shutil
from pprint import pprint


def world_to_pixel(K, rgb_matrix, destination,  curr_position):

    point_3d = np.ones((4, destination.shape[1]))
    point_3d[0] = destination[0]
    point_3d[1] = destination[1]
    point_3d[2] = curr_position[2]

    # point_3d = np.array([destination[0], destination[1], curr_position[2], 1])
    point_3d = np.round(point_3d, decimals=2)
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
    points_2d = np.round(points_2d, decimals=2)
    return points_2d


def annotate(args):

    episodes = os.listdir(args.dir)
    print(episodes)
    for episode in episodes:
        if not episode.isnumeric():
            continue
        try:
            annotation_dir = os.path.join(args.dir, episode, 'annotations')
            mask_dir = os.path.join(args.dir, episode, 'masks')
            image_dir = os.path.join(args.dir, episode, 'images')
            matrix_dir = os.path.join(args.dir, episode, 'inverse_matrix')
            if os.path.exists(mask_dir) and (len(os.listdir(mask_dir)) == len(os.listdir(image_dir)) == len(os.listdir(matrix_dir))):
                print(f"Skipping {episode}")
                # shutil.rmtree(os.path.join(args.dir, episode, 'annotations'))
                continue
            else:
                shutil.rmtree(os.path.join(args.dir, episode, 'masks'))
        except:
            pass
        os.makedirs(os.path.join(args.dir, episode,
                    'masks'), exist_ok=True)
        print(f'Episode {episode}')
        with open(os.path.join(args.dir, episode, 'vehicle_positions.txt')) as f:
            coordinates = f.readlines()
        float_coordinates = np.array([[float(x.strip()) for x in last_coordinate.split(
            ',')] for last_coordinate in coordinates])
        with open(os.path.join(args.dir, episode, 'target_positions.txt')) as f:
            coordinates = f.readlines()
        target_coordinates = np.array([[float(x.strip()) for x in last_coordinate.split(
            ',')] for last_coordinate in coordinates])[:, :3]
        target_coordinates[:-3] = target_coordinates[3:]

        print(float_coordinates.shape)

        K = np.load(os.path.join(args.dir, episode, 'camera_intrinsic.npy'))

        frames = sorted(os.listdir(os.path.join(args.dir, episode, 'images')))

        for i, frame in enumerate(frames):
            name = '.'.join(frame.split('.')[:-1])
            # print(frame, name)
            if i >= float_coordinates.shape[0]:
                break
            if i >= float_coordinates.shape[0]:
                break

            try:
                inverse_matrix = np.load(os.path.join(
                    args.dir, episode, 'inverse_matrix', name+'.npy'))
            except:
                print('NOT FOUND:', os.path.join(
                    args.dir, episode, 'inverse_matrix', name+'.npy'))

            # annotation = world_to_pixel(
            #     K, inverse_matrix, target_coordinate, relative_coords[i])
            # im = cv2.imread(os.path.join(args.dir, episode, 'images', frame))

            im = np.zeros((args.height, args.width))

            if im is None:
                break

            x_offsets = np.linspace(-2, 2, num=150)
            y_offsets = np.linspace(-2, 2, num=150)
            X, Y = np.meshgrid(x_offsets, y_offsets)

            mesh = np.dstack([X, Y])

            mesh = mesh.reshape(-1, 2)

            mesh = np.hstack([mesh, np.zeros((mesh.shape[0], 1))]).T

            annotations = world_to_pixel(
                K, inverse_matrix, target_coordinates[i].reshape(3, 1)+mesh, float_coordinates[i]).T

            for i in range(annotations.shape[0]):
                x = round(annotations[i, 0])
                y = round(annotations[i, 1])
                if x < 0 or x >= args.width or y < 0 or y >= args.height:
                    continue
                # import pdb; pdb.set_trace()
                im = cv2.circle(im, (int(x), int(y)), 4,
                                (255), thickness=-1)

            # for x_offset in np.linspace(-2, 2, num=150):
            #     for y_offset in np.linspace(-2, 2, num=150):
            #         annotation = world_to_pixel(
            #             K, inverse_matrix, target_coordinate+np.array([x_offset, y_offset, 0]), relative_coords[i])
            #         im = cv2.circle(im, (round(annotation[0]), round(
            #             annotation[1])), 2, (0, 255, 0), thickness=-1)
            cv2.imwrite(os.path.join(
                args.dir, episode, 'masks', frame), im)


def main():
    argparser = argparse.ArgumentParser(
        description='VLN data annotater')
    argparser.add_argument(
        '-d', '--dir',
        default='_out/',
        help='Input Data Directory Path')
    argparser.add_argument(
        '-n', '--height',
        default=720,
        help='input image height')
    argparser.add_argument(
        '-m', '--width',
        default=1280,
        help='input image width')
    args = argparser.parse_args()
    annotate(args)


if __name__ == '__main__':
    main()
