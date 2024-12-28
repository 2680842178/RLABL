from skimage.metrics import structural_similarity as ssim
from utils import device
import numpy
import cv2
from skimage.metrics import structural_similarity as ssim

def contrast_ssim(img1, img2):
    if img1 is None or img2 is None:
        return 0
    # return ssim(img1, img2, multichannel=True, channel_axis=2)
    if img1.shape != img2.shape:
        target_size = (min(img1.shape[0], img2.shape[0]),
                       min(img1.shape[1], img2.shape[1]))
        img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)

    # print(img1.shape, img2.shape)
    return ssim(img1, img2)

def contrast(mutation1, mutation2) -> float: # that means the similarity between two mutations
    if mutation1 is None or mutation2 is None:
        return 0
    return ssim(mutation1, mutation2, multichannel=True, channel_axis=2)

        
def obs_To_mutation(pre_obs, obs, preprocess_obss):
    pre_image_data=preprocess_obss([pre_obs], device=device).image
    image_data=preprocess_obss([obs], device=device).image
    input_tensor = image_data - pre_image_data
    input_tensor = numpy.squeeze(input_tensor)
    # input_batch = input_tensor.unsqueeze(0).permute(0, 3, 1, 2)
    return input_tensor