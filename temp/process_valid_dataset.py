# the new dataset needs to be processed using a validation file 
import random
import os
import os.path as osp
from tqdm import tqdm
import cv2
import shutil

def read_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()

    return lines

def copy_files(filepath, target_dir):
    command_format = 'scp -r gnode{}:{} {}'

    lines = read_file(filepath)

    VALIDATION_SET = 2048
    # random.shuffle(lines)

    validation_lines = lines[:VALIDATION_SET]

    source_node = '080'

    # for the validation_lines -- copy them from source to target
    for validation_line in tqdm(validation_lines):
        current_dir = validation_line.rsplit('/', 2)[1]
        current_target = osp.join(target_dir, current_dir)
        os.makedirs(current_target, exist_ok=True)

        command = command_format.format(source_node, validation_line, osp.join(current_target, osp.basename(validation_line)))
        output = os.popen(command).read()

        # print(output)

def read_frames(video_file):
    video_capture = cv2.VideoCapture(video_file)
    ret, frame = video_capture.read()
    frames = list()
    while ret:
        frames.append(frame)
        ret, frame = video_capture.read()

    return frames

def write_frames(target_dir, frames):
    for index, frame in enumerate(frames):
        framepath = osp.join(target_dir, str(index).zfill(3) + '.png')

        cv2.imwrite(framepath, frame)


def resize_frame(frame, resize_dim):
    return cv2.resize(frame, (resize_dim, resize_dim), interpolation=cv2.INTER_LINEAR)

def resize_frames(frames, resize_dim=128):
    resized_frames = list()
    for frame in frames:
        resized_frames.append(resize_frame(frame, resize_dim))

    return resized_frames

def generate_data(dir_path, save_dir, resize_dim):
    TO_SAVE = 25
    filenames = list()
    for root, _, files in os.walk(dir_path):
        for file in files:
            filename = osp.join(root, file)
            filenames.append(filename)

    print(f'Generating frames and saving to dir')
    for filename in tqdm(filenames):
        frames = read_frames(filename)[:TO_SAVE]
        resized_frames = resize_frames(frames, resize_dim)

        # write the frames to the target dir
        # target_dir = filename.rsplit('/', 1)[0] + osp.basename(filename).split('.')[0]
        target_dir = osp.join(save_dir, filename.rsplit('/', 2)[1] + osp.basename(filename).split('.')[0])
        os.makedirs(target_dir, exist_ok=True)

        # write the frames to the target_dir 
        write_frames(target_dir, resized_frames)


# use the indices file to copy the data from one folder to another 
def copy_using_indices(indices_file, validation_file, dir_path, target_dir):
    with open(indices_file, 'r') as f:
        indices = f.read().splitlines()

    # get the path corresponding to the indices in the validation_file 
    with open(validation_file, 'r') as f:
        lines = f.read().splitlines()
    
    indices = [int(index) for index in indices]

    # get the lines that correspond to the indices 
    lines_indices = list()
    lines_indices.extend([lines[indices[i]] for i in range(len(indices))])

    print(f'Copying selected dirs from source to target, dirs to copy : {len(lines_indices)}')

    for line_index in tqdm(lines_indices):
        # get the dir path 
        frames_dir = ''.join(line_index.rsplit('/', 2)[-2:]).split('.')[0]

        # full frames dir path 
        frames_dir_path = osp.join(dir_path, frames_dir)

        # copy the dirs path from current location to target location 
        shutil.copytree(frames_dir_path, osp.join(target_dir, frames_dir))


if __name__ == '__main__':
    validation_file = '/ssd_scratch/cvit/aditya1/all-except-valid-how2signfaces-valids.txt'
    target_dir = '/ssd_scratch/cvit/aditya1/inversion_styleganv_dataset_straight_256_subset'
    save_dir = '/ssd_scratch/cvit/aditya1/inversion_styleganv_dataset_straight_frames'

    indices_file = '/ssd_scratch/cvit/aditya1/indices-256.txt'

    # copy_files(validation_file, target_dir)
    resize_dim = 128
    # generate_data(target_dir, save_dir, resize_dim)
    copy_using_indices(indices_file, validation_file, save_dir, target_dir)