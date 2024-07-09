import SimpleITK as sitk


def apply_window(volume: sitk.Image, level: int, width: int) -> sitk.Image:
    window_min = level - width // 2
    window_max = level + width // 2
    return sitk.IntensityWindowing(volume,
                                   windowMinimum=window_min, windowMaximum=window_max,
                                   outputMinimum=window_min, outputMaximum=window_max)


def normalize_grayscale(volume: sitk.Image) -> sitk.Image:
    return sitk.RescaleIntensity(sitk.Cast(volume, sitk.sitkFloat32), outputMinimum=0, outputMaximum=1)


def denoise_image(volume: sitk.Image) -> sitk.Image:
    return sitk.SmoothingRecursiveGaussian(volume,
                                           sigma=[0.75] * volume.GetDimension(),
                                           normalizeAcrossScale=True)


def region_growing(volume: sitk.Image, seed_list: list) -> sitk.Image:
    return sitk.ConfidenceConnected(volume, seed_list, multiplier=2., numberOfIterations=1)


if __name__ == '__main__':
    print(f'Executed {__file__} as main')
