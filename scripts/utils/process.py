from skimage.metrics import structural_similarity as ssim

def contrast(mutation1, mutation2) -> float: # that means the similarity between two mutations
    if mutation1 is None or mutation2 is None:
        return 0
    return ssim(mutation1, mutation2, multichannel=True, channel_axis=2)

    