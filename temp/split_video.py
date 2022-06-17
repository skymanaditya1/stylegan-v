# extract 250,000 frames from a video file and save 25 frames in each dir 
# code for creating dataset from video
import cv2
import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import argparse

def write_frame(filepath, frame):
    cv2.imwrite(filepath, frame)

def write_frames(save_dir, frames):
    for index, frame in enumerate(frames):
        filepath = osp.join(save_dir, str(index).zfill(3) + '.png')

        write_frame(filepath, frame)


def read_frames(video_path):
    frames = list()
    video_stream = cv2.VideoCapture(video_path)
    ret, frame = video_stream.read()
    while ret:
        frames.append(frame)
        ret, frame = video_stream.read()

    return frames


def resize_frame(image, resize_dim):
    center = image.shape[0]/2, image.shape[1]/2

    lower_dim = min(image.shape[0], image.shape[1])

    x = center[1] - lower_dim / 2
    y = center[0] - lower_dim / 2

    cropped_image = image[int(y):int(y+lower_dim), int(x):int(x+lower_dim)]

    # resize to the dimension - 256 x 256
    resized = cv2.resize(cropped_image, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

    return resized

def resize_frames(frames, resize_dim):
    resized = list()
    for frame in frames:
        resized.append(resize_frame(frame, resize_dim))

    return resized


def opencv_resize_frame(frame, resize_dim):
    return cv2.resize(frame, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

def opencv_resize_frames(frames, resize_dim):
    resized = list()
    for frame in frames:
        resized.append(opencv_resize_frame(frame, resize_dim))

    return resized

# sample and save 25 frames from the video 
def generate_video(video_path, target_video_dir, video_threshold=1e1, video_frames_threshold=25):

    total_frames_processed = 0

    video_stream = cv2.VideoCapture(video_path)
    ret, frame = video_stream.read()

    index = 0
    
    with tqdm(total=video_threshold) as pbar:
        while index < video_threshold:

            current_frames = list()

            while ret and len(current_frames) < video_frames_threshold:
                current_frames.append(frame)
                ret, frame = video_stream.read()

                total_frames_processed += 1

            index += 1

            current_video_dir = osp.join(target_video_dir, str(index).zfill(5))
            os.makedirs(current_video_dir, exist_ok=True)
            write_frames(current_video_dir, current_frames)

            pbar.update(1)

            if not ret:
                break

    print(f'Total number of frames processed : {total_frames_processed}')

def generate_video_dataset(video_dir_path, save_dir_path, frames_to_sample=25, resize_dim=128):
    videos = glob(video_dir_path + '/*.mp4')
    for video in tqdm(videos):
        target_dir = osp.join(save_dir_path, osp.basename(video).split('.')[0])
        os.makedirs(target_dir, exist_ok=True)

        frames = read_frames(video)

        # make the call to resize the frames here
        resized = opencv_resize_frames(frames, resize_dim)

        # write the first 25 resized frames to the disk
        write_frames(target_dir, resized[:frames_to_sample])

def generate_dir_from_video(video_file, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    frames = read_frames(video_file)
    write_frames(target_dir, frames)

# video_dir_path includes all the videos
def generate_multidir_videos(video_dir_path, save_dir):

    filepaths = list()
    for root, _, files in os.walk(video_dir_path):
        files = [osp.join(root, filename) for filename in files if filename.endswith('.mp4')]
        filepaths.extend(files)

    # generate_video_dataset function gets called for all the videos
    for filepath in tqdm(filepaths):
        target_dir = save_dir + '/' + osp.join(filepath.split(video_dir_path, 1)[-1].rsplit('/', 1)[0], osp.basename(filepath).split('.')[0])

        generate_dir_from_video(filepath, target_dir)


def main(args):
    # processing code for a single video -- generate dataset from a single video
    # video_path = '/ssd_scratch/cvit/sesh/rainbow_jelly.mkv'
    # save_video_dir = '/ssd_scratch/cvit/sesh/rainbow_jelly_dataset'
    
    # os.makedirs(save_video_dir, exist_ok=True)

    # video_threshold = 1e4
    # video_frames_threshold = 25

    # print(f'Processing video : {video_path}, saving at {save_video_dir}')

    # generate_video(video_path, save_video_dir, video_threshold, video_frames_threshold)

    # generate dirs from videos 
    # video_dir_path = '/ssd_scratch/cvit/aditya1/data/mnist'
    # save_dir_path = '/ssd_scratch/cvit/aditya1/data/mnist_data'

    # generate_video_dataset(video_dir_path, save_dir_path)


    video_dir_path = args.video_dir_path
    save_dir_path = args.save_dir_path

    # video_dir_path = '/ssd_scratch/cvit/aditya1/FVDEvaluations'
    # save_dir_path = '/ssd_scratch/cvit/aditya1/FVDEvaluations_Frames'

    # generate_multidir_videos(video_dir_path, save_dir_path)

    generate_video_dataset(video_dir_path, save_dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir_path', type=str)
    parser.add_argument('--save_dir_path', type=str)
    args = parser.parse_args()

    main(args)