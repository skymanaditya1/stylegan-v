# generate the fvd scores by preparing the mocogan dataset

from genericpath import exists
import os.path as osp
import cv2
from glob import glob
from tqdm import tqdm
import os
import argparse

def read_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)

    frames = list()

    ret, frame = video_capture.read()
    while ret:
        frames.append(frame)
        ret, frame = video_capture.read()


    return frames

def write_frames(dir_path, frames):
    for index, frame in enumerate(frames):
        filename = osp.join(dir_path, str(index).zfill(3) + '.png')
        cv2.imwrite(filename, frame)


def main(dataset):
    SOURCE_DIR = '/ssd_scratch/cvit/aditya1/mocoganhd_results'
    TARGET_DIR = '/ssd_scratch/cvit/aditya1/mocoganhd_framesonly_fvd'
    # dataset = 'rainbowjelly'

    source_dir = osp.join(SOURCE_DIR, dataset)

    print(f'The source dir is : {source_dir}')

    # get all the dirs from the source_dir
    source_dirs = sorted(glob(source_dir + '/*'))

    for dirname in tqdm(source_dirs):
        # get the target dir
        target_dir = osp.join(TARGET_DIR, dataset, osp.basename(dirname))
        
        print(f'Target dir is : {target_dir}')

        # video_files = list(set(glob(target_dir + '/*.mp4')) - set(glob(target_dir + '/*_noise.mp4')))
        video_files = list(set(glob(dirname + '/*.mp4')) - set(glob(dirname + '/*_noise.mp4')))

        print(f'The length of video files : {len(video_files)}')

        for video_file in video_files:
            current_frames = read_frames(video_file)
            current_dir = osp.join(target_dir, osp.basename(video_file).split('.')[0])
            os.makedirs(current_dir, exist_ok=True)
            write_frames(current_dir, current_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    main(args.dataset)