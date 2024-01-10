import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.figure import Figure

class Simple2DSample:
    def __init__(
            self, 
            number, 
            color,
            gaussian_mean=0,
            gaussian_var=0.01,
            optical_distortion_factor=0.5
        ):
        self.number = str(number)
        self.color = color
        self.gaussian_mean = gaussian_mean
        self.gaussian_var = gaussian_var
        self.optical_distortion_factor = optical_distortion_factor
        self.img = self._text_to_rgba(self.number, color="blue", fontsize=20, dpi=200)

    def generate_light_t(self):
        return self.add_optical_distortion(self.add_gaussian_noise(self.img))

    def generate_medium_t(self):
        return self.apply_grid_distortion(self.generate_light_t())

    def generate_hard_t(self):
        return self.add_random_shift_scale_rotate(self.shuffle_rgb_channels(self.generate_medium_t()))

    def _text_to_rgba(self, s, *, dpi, **kwargs):
        fig = Figure(facecolor='none')
        fig.text(0, 0, s, **kwargs)
        with BytesIO() as buf:
            fig.savefig(buf, dpi=dpi, format='png', bbox_inches='tight', pad_inches=0.2)
            buf.seek(0)
            rgba = plt.imread(buf)
    
        # remove alpha chan
        img = (rgba * 255).astype(np.uint8)
        return img

    def add_gaussian_noise(self, img):
        row, col, ch = img.shape
        sigma = self.gaussian_var**0.5
        g = np.random.normal(self.gaussian_mean, sigma, (row, col, ch))
        g = g.reshape(row, col, ch)
        img = np.clip(g+img, 0, 1)
        return img

    def shuffle_rgb_channels(self, image):
        channels = cv2.split(image)
        np.random.shuffle(channels)
        shuffled_image = cv2.merge(channels)
        return shuffled_image

    def add_optical_distortion(self, img):
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

    def add_random_shift_scale_rotate(self, img):
        rows, cols = img.shape[:2]
        dx, dy = np.random.randint(-10, 10, 2)
        scale = np.random.uniform(0.9, 1.1)
        rotation = np.random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, scale)
        M[0, 2] += dx
        M[1, 2] += dy
        transformed_image = cv2.warpAffine(img, M, (cols, rows))
        return transformed_image

    def apply_grid_distortion(self, image, num_steps=10, distortion=1):
        height, width = image.shape[:2]
    
        # Create the original and distorted grid points
        x = np.linspace(0, width, num_steps + 1)
        y = np.linspace(0, height, num_steps + 1)
        grid_x, grid_y = np.meshgrid(x, y)
        distorted_grid_x = grid_x + np.random.normal(0, distortion, grid_x.shape)
        distorted_grid_y = grid_y + np.random.normal(0, distortion, grid_y.shape)
    
        # Interpolate the grid to create the map
        map_x = cv2.resize(distorted_grid_x, (width, height)).astype(np.float32)
        map_y = cv2.resize(distorted_grid_y, (width, height)).astype(np.float32)
    
        # Apply the distortion
        distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_REFLECT_101)
    
        return distorted_image

n = Simple2DSample(2, color='blue')
light_img = n.generate_light_t()
medium_img = n.generate_medium_t()
plt.imshow(medium_img)
plt.show()