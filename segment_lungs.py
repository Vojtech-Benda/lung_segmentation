import sys
import argparse
import SimpleITK as sitk
import matplotlib.pyplot as plt
import gui
import utils


def parse_arguments():
    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument('-f', '--filepath', help='Path to file', type=str)
    
    # optional arguments
    parser.add_argument('-l', '--level', help='CT windowing level in HU', type=int, default=0)
    parser.add_argument('-w', '--width', help='CT windowing width in HU', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    print('Executed with these arguments: ')
    for arg in args.__dict__:
        print(f'{arg}: {args.__dict__[arg]}')

    if args.width < 0:
        print(f'CT window width is negative ({args.width} < 0), making it positive to avoid errors')

    file_path = args.filepath
    ct_window_level = args.level
    ct_window_width = abs(args.width)

    try:
        loaded_vol = sitk.ReadImage(file_path)
        print(loaded_vol.GetSize(), loaded_vol.GetPixelIDTypeAsString())
    except RuntimeError as error:
        print(error)
        sys.exit()

    # preprocessing volume before segmentation
    if ct_window_width > 0:
        loaded_vol = utils.apply_window(loaded_vol, ct_window_level, ct_window_width)
    normalized_vol = utils.normalize_grayscale(loaded_vol)
    denoised_vol = utils.denoise_image(normalized_vol)

    ct_viewer = gui.CTViewer(denoised_vol)
    plt.show()
