# script used for computing the fvd scores between the real and the fake datasets 
# alternatively, it can be used for computing the fvd scores from pretrained stylegan_v checkpoints

import argparse
import os
import os.path as osp
from re import S
from sys import stderr

from numpy import extract
from tqdm import tqdm
from glob import glob
import json

# RESULTS_SAVE_DIR = '/home2/aditya1/cvit/slp/'
HOME_DIR = '/home2/aditya1/cvit/slp/fvd_results'
model = 'styleganv'

def read_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    data = dict()

    for line in lines:
        key, value = line.split('\t')
        data[key] = value

    return data

# get the fvd metric from the json file 
def extract_fvd_json(json_line):
    print(json_line)
    data = json.loads(json_line)
    return data['results']['fvd2048_16f']


# an intermediate function for saving the results - in case the training stops in between
def save_file_intermediate(filename, data, dataset):
    save_dir = osp.join(HOME_DIR, model, dataset)
    os.makedirs(save_dir, exist_ok=True)

    filename = osp.join(save_dir, osp.basename(filename).split('.')[0] + '_intermediate.txt')

    with open(filename, 'a') as f:
        for key, value in data.items():
            f.write(key + '\t' + str(value) + '\n')

# save the results to the file 
def save_file(filename, data, extra_commands, dataset, sort_data=True):

    # get the dir and create it if it doesn't exist 
    # dir_name = osp.basename(filename)
    # dir_name = osp.join(RESULTS_SAVE_DIR, filename.rsplit('/', 1)[0])
    save_dir = osp.join(HOME_DIR, model, dataset)
    os.makedirs(save_dir, exist_ok=True)
    filename = osp.join(save_dir, osp.basename(filename))

    # this code is failing for some reason
    if sort_data:
        data_sorted = {k:v for k,v in sorted(data.items(), key=lambda a : a[1])}
        data = data_sorted

    # the command which includes the checkpoint dir and the data dir also needs to be saved

    with open(filename, 'w') as f:
        for key, value in data.items():
            f.write(key + '\t' + str(value) + '\n')

        # write the additional information
        f.write(extra_commands + '\n')

    print(f'Written to file : {filename}')

# computes the fvd scores between multiple datasets
# the fake-real mapping is created using a file read as dictionary
def compute_fvd_multiple(fake_dir, real_dir, styleganv_path, mapping_file, gpu_id, mirror, result_file):
    print(f'Computing FVD for multiple directories')
    print(f'Reading file : {mapping_file}') # reads this file to get the mapping between real and fake dirs
    data = read_file(mapping_file)
    print(f'Rows read : {len(data)}')

    command_format = 'CUDA_VISIBLE_DEVICES={} python src/scripts/calc_metrics_for_dataset.py --real_data_path {} --fake_data_path {} --mirror {} --gpus 1 --resolution 128 --metrics fvd2048_16f --verbose 0 --use_cache 0'

    os.chdir(styleganv_path)
    results = dict()

    for fake_dir_path, real_dir_path in data.items():
        fake_dir_path, real_dir_path = osp.join(fake_dir, fake_dir_path), osp.join(real_dir, real_dir_path)

        # change the dir before executing the command?
        # os.chdir('/ssd_scratch/cvit/aditya1/stylegan-v')

        command = command_format.format(gpu_id, real_dir_path, fake_dir_path, mirror)
        print(f'Executing command : {command}')
        print(f'Real path : {real_dir_path}, fake path : {fake_dir_path}')
        
        output = os.popen(command).read()
        try:
            fvd_score = extract_fvd_json(output)
        except:
            fvd_score = -1
            pass

        results[fake_dir_path] = fvd_score

        # print(output)

    # save the results to the output file

    save_file(result_file, results, command)

# computes the fvd score between a single pair of datasets
def compute_fvd_single(fake_dir_path, real_dir_path, styleganv_path, gpu_id, mirror, result_file):
    command_format = 'CUDA_VISIBLE_DEVICES={} python src/scripts/calc_metrics_for_dataset.py --real_data_path {} --fake_data_path {} --mirror {} --gpus 1 --resolution 128 --metrics fvd2048_16f --verbose 0 --use_cache 0'

    # os.chdir('/ssd_scratch/cvit/aditya1/stylegan-v')
    os.chdir(styleganv_path)

    command = command_format.format(gpu_id, real_dir_path, fake_dir_path, mirror)
    print(f'Executing command : {command}')
    print(f'Real path : {real_dir_path}, fake path : {fake_dir_path}')

    output = os.popen(command).read()

    try:
        fvd_score = extract_fvd_json(output)
    except:
        fvd_score = -1
        pass

    # for a single valuation, the results could be printed on the terminal or written to a file -- defaulting to file for now
    save_file(result_file, {fake_dir_path : fvd_score}, command)

# compute the fvd scores from the pretrained checkpoint
def compute_fvd_pretrained_checkpoint(checkpoint, experimental_config, styleganv_path, gpu_id, mirror, result_file, dataset_path, return_fvd=False):
    os.chdir(styleganv_path)

    command_format = 'CUDA_VISIBLE_DEVICES={} python src/scripts/calc_metrics.py --network_pkl {} --mirror {} --gpus 1 --cfg_path {} --metrics fvd2048_16f --data {} --verbose 0 --use_cache 0'
    command = command_format.format(gpu_id, checkpoint, mirror, experimental_config, dataset_path)

    print(f'Checkpoint : {checkpoint}')
    output = os.popen(command).read()

    try:
        fvd_score = extract_fvd_json(output)
    except:
        fvd_score = -1
        pass

    if return_fvd:
        return fvd_score
    else:
        save_file(result_file, {checkpoint : fvd_score}, command) # for a single instance, the fvd_score could be displayed instead
    

# compute the metrics for a dir full of pretrained checkpoints 
def compute_fvd_pretrained_checkpoint_dir(checkpoint_dir, experimental_config, styleganv_path, gpu_id, mirror, result_file, dataset, dataset_path, debug=False, skip_checkpoint=5):
    os.chdir(styleganv_path)
    results = dict()
    checkpoints = sorted(glob(checkpoint_dir + '/*.pkl'))

    if debug:
        checkpoints = [checkpoints[0]] # this is enabled during debug 
    
        # send some dummy results 
        dummy_results = {'dummy': 1}
        save_file(result_file, dummy_results, checkpoint_dir + '\t' + styleganv_path, dataset)

    else:
        print(f'Total number of pretrained checkpoints : {len(checkpoints)}')

        should_skip = True
        if should_skip:
            print(f'Skipping checkpoints by {skip_checkpoint}')
            # this is used to compute the fvd scores for every `skip_checkpoint` checkpoint
            skipped_checkpoints = [checkpoints[i] for i in range(len(checkpoints)) if i % skip_checkpoint == 0]

            checkpoints = skipped_checkpoints

        print(f'After checkpoints : {len(checkpoints)}')

        for checkpoint in tqdm(checkpoints):
            # print(f'Not skipping any checkpoints')
            fvd_score = compute_fvd_pretrained_checkpoint(checkpoint, experimental_config, styleganv_path, gpu_id, mirror, result_file, dataset_path, return_fvd=True)

            results[checkpoint] = fvd_score

            # save to the intermediate result file after every epoch
            save_file_intermediate(result_file, {checkpoint : fvd_score}, dataset)
        
        save_file(result_file, results, checkpoint_dir + '\t' + styleganv_path, dataset)

def save_file1(filename, data, mode, sort_data=True):
    # sort the data according to value if required

    if sort_data:
        data_sorted = {k:v for k,v in sorted(data.items(), key=lambda a : a[1])}
        data = data_sorted

    # the command which includes the checkpoint dir and the data dir also needs to be saved

    with open(filename, mode) as f:
        for key, value in data.items():
            f.write(key + '\t' + str(value) + '\n')

    print(f'Written to file : {filename}')

def compute_fvd_multiple_fixed(root_dir, real_dir, styleganv_path, gpu_id, mirror, save_results):
    fake_dirs = glob(root_dir + '/*')

    os.chdir(styleganv_path)

    command_format = 'CUDA_VISIBLE_DEVICES={} \
        python src/scripts/calc_metrics_for_dataset.py \
        --real_data_path {} --fake_data_path {} --mirror {} \
        --gpus 1 --resolution 128 --metrics fvd2048_16f --verbose 0 --use_cache 0'

    results = dict()

    save_intermediate = osp.join(save_results.rsplit('/', 1)[0], osp.basename(save_results).split('.')[0] + '_intermediate.txt')

    # compute the fvd for each of the fake dirs and real dirs pair 
    for fake_dir in tqdm(fake_dirs):
        # compute the fvd between fake_dir and real_dir 
        command = command_format.format(gpu_id, real_dir, fake_dir, mirror)

        print(f'Command executed is : {command}')
        print(f'Fake dir is : {fake_dir}')

        output = os.popen(command).read()

        try:
            # get the fvd score from the output 
            fvd_score = extract_fvd_json(output)
        except:
            fvd_score = -1
            pass

        # save the intermediate results to a file
        save_file1(save_intermediate, {fake_dir : fvd_score}, 'a', sort_data=False)

        results[fake_dir] = fvd_score

    # save the full results to the file 
    save_file1(save_results, results, 'w')

# compute the FVD scores either from the dataset of generated videos 
# or from the pretrained checkpoint
def main(args):

    if args.mode == 1:
        if args.type == 1:
            compute_fvd_single(args.fake_dir, args.real_dir, args.styleganv_path, args.gpu_id, args.mirror, args.save_results)
        elif args.type == 2:
            # compute_fvd_multiple(args.fake_dir, args.real_dir, args.styleganv_path, args.mapping_file, args.gpu_id, args.mirror, args.save_results)
            compute_fvd_multiple_fixed(args.fake_dir, args.real_dir, args.styleganv_path, args.gpu_id, args.mirror, args.save_results)

    elif args.mode == 2:
        if args.type == 1:
            compute_fvd_pretrained_checkpoint(args.checkpoint, args.experimental_config, args.styleganv_path, args.gpu_id, args.mirror, args.save_results)
        elif args.type == 2:
            compute_fvd_pretrained_checkpoint_dir(args.checkpoint_dir, args.experimental_config, args.styleganv_path, args.gpu_id, args.mirror, args.save_results, args.dataset, args.dataset_path, args.debug)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--type', default=1, type=int) # 1 for single 2 for multiple
    parser.add_argument('--gpu_id', default=3, type=int)
    parser.add_argument('--mirror', default=1, type=int)
    parser.add_argument('--save_results', type=str, required=True)

    # mode 1 specifies computing the fvd using the generated videos
    parser.add_argument('--fake_dir', type=str)
    parser.add_argument('--real_dir', type=str)
    parser.add_argument('--styleganv_path', default='/ssd_scratch/cvit/aditya1/stylegan-v', type=str)
    parser.add_argument('--mapping_file', type=str)
    
    # mode 2 indicates computing the fvd using the pretrained checkpoint(s)
    # arguments required for computing from the pretrained checkpoint 
    # --pretrained_checkpoint --checkpoint_dir --dataset --experimental_config
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--checkpoint_dir', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--experimental_config', type=str)

    parser.add_argument('--dataset_path', type=str)

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)


''' 
Example usages -- 
1) Evaluating on a dir of dataset 

python compute_fvd.py --mode 1 --type 2 \
    --fake_dir /ssd_scratch/cvit/aditya1/FVDEvaluations_Frames \
    --real_dir /ssd_scratch/cvit/aditya1/fvd_eval_data \
    --mapping_file /ssd_scratch/cvit/aditya1/fake_real_mapping_tmlr.txt \
    --mirror 0
'''