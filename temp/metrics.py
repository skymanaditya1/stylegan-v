# generate the metrics 

from glob import glob 
import cv2
from skimage.metrics import structural_similarity as ssim

predictions_dir = '/home2/aditya1/cvit/slp/stylegan-v/temp/inversion/individual_frames'
output_dir = '/home2/aditya1/cvit/slp/stylegan-v/temp/inversions/BlhCuryvt88_24-8-rgb_front_face_cut-0'

prediction_paths = sorted(glob(predictions_dir + '/*.png'))
output_paths = sorted(glob(output_dir + '/*.png'))

def resize_image(image, resize_dim):
    return cv2.resize(image, (resize_dim, resize_dim), interpolation=cv2.INTER_AREA)

def read_image(path, resize_dim=128):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    resized = resize_image(image, resize_dim)

    return resized

def read_images(paths, resize_dim):
    images = list()
    for path in paths:
        images.append(read_image(path, resize_dim))

    return images

# compute ssim between two set of frames 
def compute_ssim(prediction_frames, output_frames):
    # find the smaller length between the lists
    length = len(prediction_frames) if len(prediction_frames) < len(output_frames) else len(output_frames)
    prediction_frames = prediction_frames[:length]
    output_frames = output_frames[:length]

    ssim_total = 0
    for i, (p, o) in enumerate(zip(prediction_frames, output_frames)):
        ss = ssim(p, o, multichannel=True)
        ssim_total += ss

    print(f'Avg ssim is : {ssim_total/length}')

resize_dim = 128

predictions = read_images(prediction_paths, resize_dim)
outputs = read_images(output_paths, resize_dim)

compute_ssim(predictions, outputs)