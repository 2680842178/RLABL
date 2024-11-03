from skimage.metrics import structural_similarity as ssim
from utils import device
import numpy

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