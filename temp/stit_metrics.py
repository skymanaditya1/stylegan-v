# method used for computing the TL-ID and TG-ID metrics from the STIT paper 
# TL-ID computes the temporally local identity preservation 
# TL-GD computes the temporally global identity preservation 

from concurrent.futures import process
import os

import cv2
import numpy as np
# from insightface.app import FaceAnalysis
from tqdm import tqdm
from deepface import DeepFace
from glob import glob
import argparse
import os.path as osp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

RESULTS_DIR = '/ssd_scratch/cvit/aditya1/tl_tg_id_metrics'

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory):
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory
    for file_name in sorted(os.listdir(directory)):
        if is_image_file(file_name):
            path = os.path.join(directory, file_name)
            images.append(path)
    return images


def get_face_embedding(app, image):
    detections = app.get(image)
    if len(detections) == 0:
        return None
    det = detections[0]
    embed = det['embedding']
    embed_normalized = embed / np.linalg.norm(embed)
    return embed_normalized


def measure_local_similarity(source_similarity_matrix, edit_similarity_matrix):
    relative_similarity = edit_similarity_matrix / source_similarity_matrix
    local_similarity = np.diag(relative_similarity, k=1)
    mean_local_similarity = local_similarity.mean()
    return mean_local_similarity


def measure_global_similarity(source_similarity_matrix, edit_similarity_matrix):
    relative_similarity = edit_similarity_matrix / source_similarity_matrix
    n = source_similarity_matrix.shape[0]
    off_diag = ~np.eye(n).astype(bool)
    off_diag_mean = np.mean(relative_similarity, where=off_diag)
    return off_diag_mean


def measure_metrics(source_video_files, edited_video_files):
    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    source_ds = make_dataset(source_video_files)
    edited_ds = make_dataset(edited_video_files)
    source_embeddings = []
    edited_embeddings = []
    for i, (source_path, edit_path) in enumerate(tqdm(zip(source_ds, edited_ds), total=len(source_ds), leave=False)):
        source_image = cv2.imread(source_path)
        edit_image = cv2.imread(edit_path)
        source_embedding = get_face_embedding(app, source_image)
        edited_embedding = get_face_embedding(app, edit_image)

        if source_embedding is None or edited_embedding is None:
            raise Exception(f'Failed detecting faces in frame {i} in video.')
        source_embeddings.append(source_embedding)
        edited_embeddings.append(edited_embedding)

    source_embeddings = np.stack(source_embeddings)
    edited_embeddings = np.stack(edited_embeddings)

    source_similarity_matrix = source_embeddings @ source_embeddings.T
    edit_similarity_matrix = edited_embeddings @ edited_embeddings.T
    mean_local_similarity = measure_local_similarity(source_similarity_matrix, edit_similarity_matrix)
    mean_global_similarity = measure_global_similarity(source_similarity_matrix, edit_similarity_matrix)

    return {'tl_id': mean_local_similarity, 'tg_id': mean_global_similarity}


def read_frames(videofile):
    video_capture = cv2.VideoCapture(videofile)
    frames = list()

    ret, frame = video_capture.read()
    while ret:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ret, frame = video_capture.read()

    return frames

# method to generate the embedding for the read frame 
def get_embedding(frame):
    model_name = 'ArcFace'
    embedding = DeepFace.represent(img_path=frame, model_name=model_name, enforce_detection=False)

    return embedding

# compute the metrics for a single pair of videos 
def compute_metrics_video(source_file, inverted_file):
    source_frames = read_frames(source_file)
    inverted_frames = read_frames(inverted_file)

    source_embeddings = list()
    inverted_embeddings = list()

    for i, (original_frame, inverted_frame) in enumerate(zip(source_frames, inverted_frames)):
        # source_embedding = get_face_embedding(app, original_frame)
        # inverted_embedding = get_face_embedding(app, inverted_frame)

        source_embedding = get_embedding(original_frame)
        inverted_embedding = get_embedding(inverted_frame)

        if source_embedding is None or inverted_embedding is None:
            raise Exception(f'Failed detecting faces in frame {i} in video.')

        source_embeddings.append(source_embedding)
        inverted_embeddings.append(inverted_embedding)

    source_embeddings = np.stack(source_embeddings)
    inverted_embeddings = np.stack(inverted_embeddings)

    source_similarity_matrix = source_embeddings @ source_embeddings.T
    inverted_similarity_matrix = inverted_embeddings @ inverted_embeddings.T
    mean_local_similarity = measure_local_similarity(source_similarity_matrix, inverted_similarity_matrix)
    mean_global_similarity = measure_global_similarity(source_similarity_matrix, inverted_similarity_matrix)

    return {'tl_id': mean_local_similarity, 'tg_id': mean_global_similarity}

def write_file(filename, data, mode='w'):
    print(f'Writing to file : {filename}')
    with open(filename, mode) as f:
        f.write('\t'.join(str(data)) + '\n')

# measure metrics for a pair of source and target video 
def measure(video_dir, img_dimension = 128):
    # this is done to skip the val path
    result_file = osp.join(RESULTS_DIR, video_dir.rsplit('/', 2)[1] + '.txt')
    intermediate_file = osp.join(RESULTS_DIR, video_dir.rsplit('/', 2)[1] + '_intermediate.txt')

    # result_file = osp.join(RESULTS_DIR, osp.basename(video_dir) + '.txt')
    # intermediate_file = osp.join(RESULTS_DIR, osp.basename(video_dir) + '_intermediate.txt')

    print(f'Starting TL-ID and TG-ID metric computation')
    global_tl_id = list()
    global_tg_id = list()
    source_files = sorted(glob(video_dir + '/*.mp4'))
    inverted_files = [source_file.replace('original', 'pred') for source_file in source_files]

    # generate the frames for both the files 
    for index, (source_file, inverted_file) in enumerate(tqdm(zip(source_files, inverted_files))):
        # compute the metrics between the source and inverted videos
        metrics = compute_metrics_video(source_file, inverted_file)
        tl_id, tg_id = metrics['tl_id'], metrics['tg_id']
        global_tl_id.append(tl_id)
        global_tg_id.append(tg_id)

        print(f'Video file : {source_file}, local and global metrics are : {tl_id, tg_id}')

        data = [source_file, tl_id, tg_id]

        # append the local scores to an intermediate file
        write_file(intermediate_file, data, 'a')

    # compute the averaged metrics 
    avg_tl_id = sum(global_tl_id)/len(global_tl_id)
    avg_tg_id = sum(global_tg_id)/len(global_tg_id)

    print(f'Total metrics computed : {len(global_tl_id)} for {len(inverted_files)} files')
    print(f'Avg tl_id : {avg_tl_id}, avg tg_id : {avg_tg_id}')

    # write the full metrics to the results file 
    with open(result_file, 'w') as f:
        f.write(f'Total metrics computed : {len(global_tl_id)} for {len(inverted_files)} files')
        f.write(f'Avg tl_id : {avg_tl_id}, avg tg_id : {avg_tg_id}')


def process_video_dirs(root_dir):
    video_dirs = glob(root_dir + '/*')

    # total number of dirs to process
    print(f'Total dirs to process : {len(video_dirs)}')

    for video_dir in video_dirs:
        measure(osp.join(video_dir, 'val'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str)
    args = parser.parse_args()

    img_dimension = 128

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
    # app.prepare(ctx_id=0, det_size=(img_dimension, img_dimension))

    video_dir = args.video_dir
    # measure(video_dir, app, img_dimension)
    process_video_dirs(video_dir)

