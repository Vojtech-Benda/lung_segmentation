import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import utils


class CTViewer(object):
    def __init__(self, volume: sitk.Image) -> None:
        self.volume = volume
        self.current_displaying_volume = volume
        self.vol_width, self.vol_height, self.vol_depth = volume.GetSize()  # x, y, z size
        self.displaying_volume = sitk.Cast(sitk.RescaleIntensity(volume), sitk.sitkUInt8)  # volume to display as RGB
        self.zaxis_index = self.vol_depth // 2
        self.current_seeds = []
        self.segmented_regions = []
        self.region_cmaps = []
        self.region_num = 1
        self.drawing_enabled = False

        self.fig, self.ax = plt.subplots(1, 1)

        array_data = sitk.GetArrayFromImage(self.volume[..., self.zaxis_index])
        self.ax.imshow(array_data, cmap='gray')

        self.fig.canvas.mpl_connect('scroll_event', self.scroll_zaxis)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_button_press)
        self.fig.canvas.draw_idle()
        print('Press D to start segmenting')

    def update_display(self):
        num_of_channels = self.current_displaying_volume.GetNumberOfComponentsPerPixel()
        array_data = sitk.GetArrayFromImage(self.current_displaying_volume[..., self.zaxis_index])

        # grayscale or rgb colormap depending on number of channels
        cmap = 'gray' if num_of_channels == 1 else 'viridis'
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()

        self.ax.imshow(array_data, cmap=cmap)

        for i, point in enumerate(self.current_seeds):
            if self.zaxis_index == point[2]:
                self.ax.scatter(point[0], point[1], s=60, marker='+', color='red')

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.fig.canvas.draw_idle()

    def on_mouse_click(self, event):
        # point at x, y, z - z is superior-inferior direction
        if event.inaxes == self.ax and self.drawing_enabled:
            self.current_seeds.append((int(round(event.xdata)),
                                       int(round(event.ydata)),
                                       self.zaxis_index))

            self.update_display()

    def scroll_zaxis(self, event):
        self.zaxis_index += int(event.step)

        # check z axis bounds (superior-inferior direction)
        if self.zaxis_index > self.vol_depth - 1:
            self.zaxis_index = self.vol_depth - 1
        elif self.zaxis_index < 0:
            self.zaxis_index = 0

        self.update_display()

    def on_button_press(self, event):
        match event.key:
            case 'd':
                if self.drawing_enabled:
                    self.drawing_enabled = False  # disable drawing if it's enabled
                    return None
                else:
                    self.drawing_enabled = True  # enable drawing if it's disabled
                self.select_seeds()

            case 'enter':
                if len(self.current_seeds) != 0:
                    segm_region = utils.region_growing(self.volume, self.current_seeds) * self.region_num
                    self.segmented_regions.append(segm_region)
                    self.region_num += 1
                    self.current_seeds.clear()
                    self.select_seeds()
                else:
                    print('No seed points selected, continue selecting or press Esc to cancel segmentation')
                    return None

                labeled_volume = sitk.Cast(sitk.NaryAdd(self.segmented_regions), sitk.sitkLabelUInt8)
                self.region_cmaps += np.random.Generator.integers(low=0, high=255, size=3, endpoint=True).tolist()

                # add label contours for each region with randomly generated rgb color
                self.current_displaying_volume = sitk.LabelMapOverlay(labeled_volume,
                                                                      self.displaying_volume,
                                                                      colormap=self.region_cmaps[::-1])
                self.update_display()

            case 'esc':
                self.drawing_enabled = False
                return None

    def select_seeds(self):
        print(f'Select seed points for region {self.region_num}\n'
              'Press Enter to finish selecting, press Esc to cancel segmentation')


if __name__ == '__main__':
    print(f'Executed file {__file__}')
