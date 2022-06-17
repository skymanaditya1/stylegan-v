# file for copying the checkpoints from a directory in a gnode to another directory in another gnode
import argparse
import os
import os.path as osp

from tqdm import tqdm

def read_file(file, to_save=5):
    with open(file, 'r') as f:
        lines = f.read().splitlines()

    # return the path of the top to_save lines 
    ckpts = [line.split('\t')[0] for line in lines[:to_save]]

    return ckpts

def copy_ckpts(source_node, target_node, target_dir, file, copy_checkpoints=True, copy_jpg_files=True):
    checkpoints = read_file(file)

    # copy_command = 'scp -r gnode{}:{}/{} gnode{}:{}'
    copy_command = 'scp -r gnode{}:{} gnode{}:{}'

    if copy_checkpoints:
        for checkpoint in tqdm(checkpoints):
            command = copy_command.format(source_node, checkpoint, target_node, target_dir)
            print(f'Executing command : {command}')
            os.system(command)

    if copy_jpg_files:
        # get the jpg path from the checkpoint 
        for checkpoint in checkpoints:
            jpg_path = osp.join(checkpoint.rsplit('/', 1)[0], 'fakes' + osp.basename(checkpoint).rsplit('-', 1)[1].split('.')[0] + '.jpg')
            # copy the jpg path if it exists - not sure how to check the existence of the jpg path on another node 
            command = copy_command.format(source_node, jpg_path, target_node, target_dir)
            print(f'Executing command : {command}')
            os.system(command)

def main(args):
    source_node = args.source_node
    target_node = args.target_node
    target_dir = args.target_dir
    file = args.file

    copy_ckpts(source_node, target_node, target_dir, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_node', type=str, required=True)
    parser.add_argument('--target_node', type=str, required=True)
    # parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--file', type=str, required=True)

    args = parser.parse_args()
    main(args)