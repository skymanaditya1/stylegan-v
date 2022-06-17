# this file is used for processing the fvd metrics and the network snapshot from the saved txt file
import argparse
import json

def save_file(data, filename, sort_data=True):

    # sort the results before saving them
    if sort_data:
        data_sorted = {k:v for k,v in sorted(data.items(), key=lambda x : x[1])}
        data = data_sorted

    # the command which includes the checkpoint dir and the data dir also needs to be saved

    with open(filename, 'w') as f:
        for key, value in data.items():
            f.write(key + '\t' + str(value) + '\n')

    print(f'Saving results in : {filename}')

def read_metrics(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    # get the fvd metrics and the corresponding snapshot from the lines 
    results_dict = dict()
    for line in lines:
        if line.startswith('{"results'):
            # conver the line to json and extract the fvd score and snapshot 
            json_data = json.loads(line)
            fvd_metrics = json_data['results']['fvd2048_16f']
            snapshot = json_data['snapshot_pkl']

            results_dict[snapshot] = fvd_metrics

    return results_dict


def read_metrics1(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    result_dict = dict()

    print(f'Number of lines read : {len(lines)}')

    for line in lines:
        key, value = line.split('\t')
        result_dict[key] = value

    return result_dict

def main(args):
    filename = args.input_file
    save_filename = args.output_file
    # results_dict = read_metrics(filename)
    # save_file(results_dict, save_filename)

    results_dict = read_metrics1(filename)
    save_file(results_dict, save_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()
    main(args)