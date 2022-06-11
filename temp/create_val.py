
import os
import os.path as osp

from glob import glob
from tqdm import tqdm
import shutil

import cv2

def save_file(filepath, data):
    with open(filepath, 'w') as f:
        for line in data:
            f.write(line + '\n')

# read from the valid file create by Bipasha and add to the file if required
def create_validation_file(current_validation_filepath, updated_validation_filepath, dataset_dir, dataset_type=2):
    print(f'Creating validation file')
    frame_threshold = 25
    video_threshold = 1e4

    with open(current_validation_filepath) as f:
        lines = f.read().splitlines()

    if dataset_type == 1:
        lines = [line.split('/')[-1] for line in lines]

        updated_valids = list()
        index = 0

        while index < len(lines) and len(updated_valids) < video_threshold:
            # get the number of frames in the current dir and check if greater than threshold
            frames = glob(osp.join(dataset_dir, lines[index]) + '/*.jpg')
            if len(frames) >= frame_threshold and lines[index][0] != '-':
                updated_valids.append(lines[index])

            index += 1
    elif dataset_type == 2:
        lines = ['/'.join(line.rsplit('/', 2)[-2:]) for line in lines]

        updated_valids = list()
        index = 0

        while index < len(lines) and len(updated_valids) < video_threshold:
            # get the number of frames inside the current dir and check if its greater than threshold
            frames = glob(osp.join(dataset_dir, lines[index]) + '/*.jpg')
            if len(frames) >= frame_threshold and lines[index][0] != '-':
                updated_valids.append(lines[index])

            index += 1

    # save the updated valids file 
    print(f'Saving {len(updated_valids)} to {updated_validation_filepath}')
    print(f'Validation file {updated_validation_filepath} created successfully')

    save_file(updated_validation_filepath, updated_valids)


# return the number of frames in the current video 
def get_video_frames(video_path):
    video_stream = cv2.VideoCapture(video_path)

    return int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))

def create_validation_file_videos(current_validation_filepath, updated_validation_filepath, dataset_dir):
    print(f'Creating validation file')
    frame_threshold = 25
    video_threshold = 1e4

    with open(current_validation_filepath) as f:
        lines = f.read().splitlines()

        lines = ['/'.join(line.rsplit('/', 2)[-2:]) for line in lines]

        updated_valids = list()
        index = 0

        while index < len(lines) and len(updated_valids) < video_threshold:
            # get the number of frames from the video 
            video_filepath = osp.join(dataset_dir, lines[index])
            frames = get_video_frames(video_filepath)
            if frames >= frame_threshold and lines[index] != '-':
                updated_valids.append(lines[index]) # adds the relative video path

            index += 1

    # save the updated valids file 
    print(f'Saving {len(updated_valids)} to {updated_validation_filepath}')
    print(f'Validation file {updated_validation_filepath} created successfully')

    save_file(updated_validation_filepath, updated_valids)


def read_frames(frame_paths):
    frames = list()
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)

        frames.append(frame)

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

# resize frames using opencv resize without aspect ratio consideration
def resize_frames_opencv(frames, resize_dim):
    resized = list()
    for frame in frames:
        resized.append(resize_frame_opencv(frame, resize_dim))

# resize frames using opencv resize without consideration of aspect ratio
def resize_frame_opencv(image, resize_dim):
    resized = cv2.resize(image, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)
    return resized


# method to flip the frame
def flip_frame(frame, flip_code):
    return cv2.flip(frame, flip_code)


# method to flip the frames
def flip_frames(frames, flip_code=0):
    flipped = list()
    for frame in frames:
        flipped.append(flip_frame(frame, flip_code))

    return flipped

def write_frame(filename, frame):
    cv2.imwrite(filename, frame)

def write_frames(frames, target_dir, dirname, dataset_type):
    if dataset_type == 1:
        current_target_dir = osp.join(target_dir, dirname)

    elif dataset_type == 2:
        current_target_dir = osp.join(target_dir, '_'.join(dirname.split('/')))

    elif dataset_type == 3:
        current_target_dir = osp.join(target_dir, '_'.join(dirname.split('/'))).split('.')[0]

    os.makedirs(current_target_dir, exist_ok=True)

    # save the frames inside the current_target_dir
    for index, frame in enumerate(frames):
        filename = osp.join(current_target_dir, str(index).zfill(3) + '.jpg')
        write_frame(filename, frame)


def create_stylegan_data(validation_filepath, dataset_dir, stylegan_dir, dataset_type=2, image_res=128):
    frame_threshold = 25
    os.makedirs(stylegan_dir, exist_ok=True)

    with open(validation_filepath, 'r') as f:
        lines = f.read().splitlines()

    print(f'Creating the stylegan directory for the validation dir')
    print(f'Stylegan data dir : {stylegan_dir}')

    for line in tqdm(lines):
        current_dir = osp.join(dataset_dir, line)
        current_frames = sorted(glob(current_dir + '/*.jpg'))[:frame_threshold]

        # read frames 
        frames = read_frames(current_frames)

        # resize frames by applying center crop
        resized = resize_frames(frames, resize_dim=image_res)

        # write the frames at the target dir 
        write_frames(resized, stylegan_dir, line, dataset_type)


# creates stylegan_data dir from input dir
def create_stylegan_data_dir(input_dir, stylegan_dir, image_res=128):
    # create the stylegan_dir 
    os.makedirs(stylegan_dir, exist_ok=True)

    input_dirs = glob(input_dir + '/*')
    print(f'Processing {len(input_dirs)} directories')
    for current_dir in tqdm(input_dirs):

        # target_dir = osp.join(stylegan_dir, osp.basename(current_dir))
        # os.makedirs(target_dir, exist_ok=True)

        current_frames = sorted(glob(current_dir + '/*.jpg'))
        frames = read_frames(current_frames)

        # resize the frames without consideration to the aspect ratio 
        resized = resize_frames(frames, resize_dim=image_res)

        write_frames(resized, stylegan_dir, osp.basename(current_dir), 1)


def read_video_frames(video_path):
    video_stream = cv2.VideoCapture(video_path)
    frames = list()

    ret, frame = video_stream.read()

    while ret:
        frames.append(frame)
        ret, frame = video_stream.read()

    return frames

# create stylegan-v data from videos 
def create_stylegan_data_videos(validation_filepath, dataset_dir, stylegan_dir, image_resize=128):

    frame_threshold = 25
    os.makedirs(stylegan_dir, exist_ok=True)

    with open(validation_filepath, 'r') as f:
        lines = f.read().splitlines()

    print(f'Creating the stylegan directory for the validation dir')
    print(f'Stylegan data dir : {stylegan_dir}')

    for line in tqdm(lines):
        current_video = osp.join(dataset_dir, line) # path of the current video itself
        frames = read_video_frames(current_video)[:frame_threshold]

        # resize frames by applying center crop
        resized = resize_frames(frames, resize_dim=image_resize)

        # write the frames at the target dir 
        write_frames(resized, stylegan_dir, line, 3)


# method to generate dir of frames from videos, without changing the resolution of the images
def generate_dir_from_videos(dir_path, stylegan_dir):

    os.makedirs(stylegan_dir, exist_ok=True)
    print(f'Saving frames in dir : {stylegan_dir}')

    dataset_type = 1
    videos = glob(dir_path + '/*.mp4')
    for video in tqdm(videos):
        frames = read_video_frames(video)
        # write the frames to dir

        # save the frames in the path specified
        write_frames(frames, stylegan_dir, osp.basename(video).split('.')[0], dataset_type)

    print(f'Frames written succesfully')


# get a list of all the files 
def generate_files_dir(dir_path, stylegan_dir):
    os.makedirs(stylegan_dir, exist_ok=True)
    print(f'Saving output frames in dir : {stylegan_dir}')

    mp4_files = list()

    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.mp4'):
                mp4_files.append(osp.join(root, file))

    print(f'Total files : {len(mp4_files)}')

    resize_dim = 128
    dataset_type = 3

    for video in tqdm(mp4_files):
        # read the frames
        frames = read_video_frames(video)

        resized = resize_frames(frames, resize_dim)

        # save the resized frames to the output dir 
        write_frames(resized, stylegan_dir, video, dataset_type)


# dir_path consists of dirs of images
# flip every image and write in outdir_pat
def flip_image_dir(dir_path, outdir_path):
    os.makedirs(outdir_path, exist_ok=True)
    input_dirs = glob(dir_path + '/*')
    for current_dir in tqdm(input_dirs):
        # read the frames in the current dir
        frame_paths = sorted(glob(current_dir + '/*.jpg'))
        frames = read_frames(frame_paths)

        flipped = flip_frames(frames)

        # save the flipped frames to the target dir 
        write_frames(flipped, outdir_path, osp.basename(current_dir), 1)

    print(f'Frames written successfully')


# sample the top 25 frames from each of the directories specified inside a main dir 
def sample_topk(dir_path, new_dir_path, topk=25):
    dirs = glob(dir_path + '/*')
    print(f'Transferring frames from {dir_path} to {new_dir_path}')
    for dirname in tqdm(dirs):
        # get all the frame paths in the current dir 
        current_frames = sorted(glob(dirname + '/*.jpg'))[:topk]
        # transfer the top k files to another dir inside the new_dir_path
        dirname = dirname.split('voxceleb_mp4')[1]
        target_dir = osp.join(new_dir_path, dirname)
        # print(f'Target dir path is : {target_dir}')
        os.makedirs(target_dir, exist_ok=True)

        for current_frame in current_frames:
            shutil.copy(current_frame, target_dir)


def main():

    # configuration setting for ucf101 on gnode61
    # input_size = 256
    # dataset_type = 1
    # current_validation_filepath = '/ssd_scratch/cvit/aditya1/stylegan-v/temp/ucf_old.txt'
    # dataset_dir = '/ssd_scratch/cvit/aditya1/baselines/ucf101'
    # styleganv_dataset_dir = '/ssd_scratch/cvit/aditya1/stylegan-v/data'
    # updated_validation_filepath = '/ssd_scratch/cvit/aditya1/stylegan-v/temp/ucf_updated.txt'

    # configuration setting for skytimelapse on gnode33

    # input_size = 128
    # dataset_type = 2

    # current_validation_filepath = '/ssd_scratch/cvit/aditya1/stylegan-v/temp/sky-timelapse.txt'
    # dataset_dir = '/ssd_scratch/cvit/timelapse/sky_timelapse/sky_train'
    # styleganv_dataset_dir = '/ssd_scratch/cvit/aditya1/stylegan-v/data/sky_train'
    # updated_validation_filepath = '/ssd_scratch/cvit/aditya1/stylegan-v/temp/sky-timelapse_updated.txt'


    # configuration setting for the how2sign hand gesture dataset

    # input_size = 128
    
    # current_validation_filepath = '/ssd_scratch/cvit/aditya1/stylegan-v/rhands-how2sign.txt'
    # dataset_dir = '/ssd_scratch/cvit/rhands_intervals_2s'
    # styleganv_dataset_dir = '/ssd_scratch/cvit/aditya1/stylegan-v/hand_gesture'
    # updated_validation_filepath = '/ssd_scratch/cvit/aditya1/stylegan-v/hand_gesture.txt'

    # create_validation_file_videos(current_validation_filepath, updated_validation_filepath, dataset_dir)

    # # read the first 25 frames from the selected dirs and resize them and copy to the target dir 

    # create_stylegan_data_videos(updated_validation_filepath, dataset_dir, styleganv_dataset_dir, image_resize=input_size)

    # configuration for the rainbow jelly dataset

    # input_size = 128

    # # read through the images in the dataset_dir, resize them, and keep them inside the stylegan_v dir
    # dataset_dir = '/ssd_scratch/cvit/aditya1/rainbow_jelly_dataset'
    # styleganv_dataset_dir = '/ssd_scratch/cvit/aditya1/stylegan-v/data/rainbow_jelly_custom'

    # create_stylegan_data_dir(dataset_dir, styleganv_dataset_dir, input_size)
    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_how2sign_videos_256'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_how2sign_256'

    # generate_dir_from_videos(dir_path, stylegan_dir_path)

    # how2sign dataset -- randomly sampled
    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_how2sign_randomsamples_videos_100'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_how2sign_randomsamples_100'

    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_how2sign_videosonly_256'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_how2sign_256'

    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_rainbowjelly_videos_128'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/inrv_predictions_rainbowjelly_128'

    # method for flipping the output generated by the pretrained checkpoint 
    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/styleganv_predictions_how2sign_videos_256'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/styleganv_predictions_how2sign_256'

    # generate dir from videos 
    # generate_dir_from_videos(dir_path, stylegan_dir_path)

    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/styleganv_predictions_how2sign_256'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/styleganv_predictions_how2sign_flipped_256'

    # flip_image_dir(dir_path, stylegan_dir_path)  

    # # preprocessing required for the voxceleb dataset
    # dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/voxceleb_mp4'
    # stylegan_dir_path = '/ssd_scratch/cvit/aditya1/stylegan-v/data/voxceleb_styleganv'

    # generate_files_dir(dir_path, stylegan_dir_path)

    # sample the top k files from the sample dir to the target dir 
    sample_dir = '/ssd_scratch/cvit/aditya1/stylegan-v/data/voxceleb_styleganv'
    target_dir = '/ssd_scratch/cvit/aditya1/stylegan-v/data/voxceleb_styleganv_sampled_top25'

    os.makedirs(target_dir, exist_ok=True)

    sample_topk(sample_dir, target_dir, topk=25)


if __name__ == '__main__':
    main()