# this is the model which takes in a series of snapshot files as input and generates the inference 
import os
from glob import glob
import os.path as osp

styleganv_dir = '/ssd_scratch/cvit/aditya1/stylegan-v'

def generate_videos(gpu_id, checkpoint, results_dir, num_videos=2):
    command_format = 'CUDA_VISIBLE_DEVICES={} python src/scripts/generate.py \
            --network_pkl {} \
            --num_videos {} --save_as_mp4 true \
            --fps 25 --video_len 128 --batch_size 25 \
            --outdir {} --truncation_psi 0.9'

    command = command_format.format(gpu_id, checkpoint, num_videos, results_dir)

    print(f'Running command : {command}')

    output = os.popen(command).read()

    print(output)

def read_checkpoints(checkpoint_dir):
    # reads all checkpoints from the dir 
    checkpoints = glob(checkpoint_dir + '/*.pkl')

    return checkpoints

def main():
    input_dir = '/ssd_scratch/cvit/aditya1/important_checkpoints/styleganv/rainbowjelly'
    gpu_id = 1
    num_videos = 2

    os.chdir(styleganv_dir)

    # results dir would depend on the checkpoint
    checkpoints = read_checkpoints(input_dir)
    for checkpoint in checkpoints:
        results_dir = osp.join(input_dir, osp.basename(checkpoint).split('.')[0])
        os.makedirs(results_dir, exist_ok=True)
        generate_videos(gpu_id, checkpoint, results_dir, num_videos)

if __name__ == '__main__':
    main()