import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.figure import Figure
from sklearn.cluster import KMeans

class Simple2DSample:
    def __init__(
            self, 
            number, 
            color,
            facecolor='red',
            fontsize=20,
            gaussian_mean=0,
            gaussian_var=0,
            optical_distortion_factor=0.5,
            grid_distortion_factor=1
        ):
        self.number = str(number)
        self.color = color
        self.fontsize = fontsize
        self.gaussian_mean = gaussian_mean
        self.gaussian_var = gaussian_var
        self.optical_distortion_factor = optical_distortion_factor
        self.grid_distortion_factor = grid_distortion_factor
        self.img = self._text_to_rgba(self.number, color="blue", fontsize=self.fontsize, dpi=200)

    def generate_light_t(self, image=None):
        return self._add_optical_distortion(image if image else self._add_g_noise(self.img))

    def generate_medium_t(self, image=None):
        return self._apply_grid_distortion(image if image else self.generate_light_t())

    def generate_hard_t(self, image=None):
        return self._add_random_shift_scale_rotate(self._shuffle_channels(image if image else self.generate_medium_t()))

    def _text_to_rgba(self, s, *, dpi, **kwargs):
        fig = Figure(facecolor='black')
        fig.text(0, 0, s, bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=1'), **kwargs)
        with BytesIO() as buf:
            fig.savefig(buf, dpi=dpi, format='png', bbox_inches='tight', pad_inches=0.07)
            buf.seek(0)
            rgba = plt.imread(buf)
    
        # remove alpha channnel
        rgba = rgba[:, :, :3]
        img = (rgba * 255).astype(np.uint8)
        return img

    def _add_g_noise(self, img):
        row, col, ch = img.shape
        sigma = self.gaussian_var**0.5
        g = np.random.normal(self.gaussian_mean, sigma, (row, col, ch))
        g = g.reshape(row, col, ch)
        img = np.clip(g+img, 0, 1)
        return img

    def _shuffle_channels(self, image):
        channels = cv2.split(image)
        channels_list = list(channels)
        np.random.shuffle(channels_list)
        shuffled_image = cv2.merge(channels_list)
        return shuffled_image

    def _add_optical_distortion(self, img):
        height, width = img.shape[:2]
        camera_matrix = np.array([[width, 0, width / 2],
                                  [0, height, height / 2],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([
            np.random.uniform(-1, 1), 
            np.random.uniform(-1, 1), 
            np.random.uniform(-0.1, 0.1), 0, 0], dtype=np.float32)
        distorted_image = cv2.undistort(img, camera_matrix, dist_coeffs)
        return distorted_image

    def _add_random_shift_scale_rotate(self, img):
        rows, cols = img.shape[:2]
        dx, dy = np.random.randint(-10, 10, 2)
        scale = np.random.uniform(0.9, 1.1)
        rotation = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        transformed_image = cv2.warpAffine(img, M, (cols, rows))
        return transformed_image

    def _apply_grid_distortion(self, image, num_steps=10):
        height, width = image.shape[:2]
    
        # Create the original and distorted grid points
        x = np.linspace(0, width, num_steps + 1)
        y = np.linspace(0, height, num_steps + 1)
        grid_x, grid_y = np.meshgrid(x, y)
        distorted_grid_x = grid_x + np.random.normal(0, self.grid_distortion_factor, grid_x.shape)
        distorted_grid_y = grid_y + np.random.normal(0, self.grid_distortion_factor, grid_y.shape)
    
        # Interpolate the grid to create the map
        map_x = cv2.resize(distorted_grid_x, (width, height)).astype(np.float32)
        map_y = cv2.resize(distorted_grid_y, (width, height)).astype(np.float32)
    
        # Apply the distortion
        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT_101)
    
        return distorted_image
    
def convert_bg(img, color=[255, 255, 255]):
    
    if not img.dtype == np.uint8:
        img = (img * 255).astype(np.uint8)
    
    black_pixels = np.where(
        (img[:, :, 0] < 5) & 
        (img[:, :, 1] < 5) & 
        (img[:, :, 2] < 5)
    )

    img[black_pixels] = color

    return img

def generate_sample():
    random_jersey_number_1 = str(np.random.randint(0, 9))
    n = Simple2DSample(random_jersey_number_1, color=colors[i%len(colors)], fontsize=20, grid_distortion_factor=1)
    light_img_1 = n.generate_light_t()
    medium_img_1 = n.generate_medium_t()
    random_jersey_number_2 = str(np.random.randint(0, 9))
    n = Simple2DSample(random_jersey_number_2, color=colors[i%len(colors)], fontsize=20, grid_distortion_factor=1)
    light_img_2 = n.generate_light_t()
    medium_img_2 = n.generate_medium_t()

    light_img = np.hstack((light_img_1, light_img_2))
    medium_img = np.hstack((medium_img_1, medium_img_2))
    hard_img = n._add_random_shift_scale_rotate(n._shuffle_channels(medium_img))
    return convert_bg(light_img), convert_bg(medium_img), convert_bg(convert_bg(hard_img)) 

if __name__ == '__main__':
    colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
    for i in range(100):
        try:
            l, m, h = generate_sample()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(l)
            ax1.set_title('light')
            ax2.imshow(m)
            ax2.set_title('medium')
            ax3.imshow(h)
            ax3.set_title('hard')
            plt.pause(10)
            plt.clf()
            plt.cla()
        except KeyboardInterrupt:
            break