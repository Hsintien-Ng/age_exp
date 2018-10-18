"""
Bio-Inspired Feature Extraction

Python implementation of Bio-Inspired Features (BIF) as described in the paper:

G. Guo, Guowang Mu, Y. Fu and T. S. Huang,
"Human age estimation using bio-inspired features",
2009 IEEE Conference on Computer Vision and Pattern Recognition, Miami, FL, 2009, pp. 112-119
http:// ieeexplore.ieee.org/document/5206681
"""
import numpy as np
import math
from utils.figure import integralImage


class BIF:
    def __init__(self):
        self.cell_sizes = np.asarray([6, 8, 10, 12, 14, 16, 18, 20])
        self.gabor_sizes = np.asarray([[5, 7], [9, 11], [13, 15], [17, 19],
                                       [21, 23], [25, 27], [29, 31], [33, 35]])
        self.gabor_sigmas = np.asarray([[2.0, 2.8], [3.6, 4.5], [5.4, 6.3], [7.3, 8.2],
                                        [9.2, 10.2], [11.3, 12.3], [13.4, 14.6], [15.8, 17.0]])
        self.gabor_wavelengths = np.asarray([[2.5, 3.5], [4.6, 5.6], [6.8, 7.9], [9.1, 10.3],
                                             [11.5, 12.7], [14.1, 15.4], [16.8, 18.2], [19.7, 21.2]])
        self.gabor_gamma = 0.3


    def bif(self, img, bands, rotations):
        """
        main function of calculating BIF
        :return: output vector of BIF
        """
        self.img = img
        self.bands = bands
        self.rotations = rotations
        assert isinstance(img, np.ndarray)
        self.img = np.asarray(self.img, dtype=np.float32)
        assert isinstance(self.bands, int), 'parameter bands must be int type'
        assert self.bands <= 8 and self.bands > 0, 'bands must be in (0, 8]'
        assert self.rotations > 0, 'rotations must be greater than 0'

        Filter_Bank = self.init_gabor_filter_bank(self.bands, self.rotations)
        Features, features_dims = self.apply_filter_bank(self.img, Filter_Bank)
        out_vec = self.Get_output_BIF(Features, features_dims)

        return out_vec


    def init_gabor_filter_bank(self, bands, rotations):
        """
        function of initializing gabor filter bank
        :return: Filter_Bank (list) each element in list should be a Filter_Unit (also list) 
        """
        Filter_Bank = []
        for r in range(rotations):
            theta = math.pi / rotations * r
            for b in range(bands):
                kernel_1 = self.get_gabor_kernel(self.gabor_sizes[b, 0], self.gabor_sizes[b, 0],
                                                 self.gabor_sigmas[b, 0], theta,
                                                 self.gabor_wavelengths[b, 0], self.gabor_gamma, 0)
                kernel_2 = self.get_gabor_kernel(self.gabor_sizes[b, 1], self.gabor_sizes[b, 1],
                                                 self.gabor_sigmas[b, 1], theta,
                                                 self.gabor_wavelengths[b, 1], self.gabor_gamma, 0)

                kernel_1 = kernel_1 / (2 * self.gabor_sigmas[b, 0] * self.gabor_sigmas[b, 0] / self.gabor_gamma)
                kernel_2 = kernel_2 / (2 * self.gabor_sigmas[b, 1] * self.gabor_sigmas[b, 1] / self.gabor_gamma)

                Filter_Unit = [kernel_1, kernel_2, self.cell_sizes[b]]
                Filter_Bank.append(Filter_Unit)

        return Filter_Bank


    def get_gabor_kernel(self, height, width, sigma, theta, Lambda, gamma, psi):
        """
        OpenCV cv:getGaborKernel to Python get_garbor_kernel Conversion
        :param height: height of the filter returned
        :param width: width of the filter returned
        :param sigma: standard deviation of the gaussian envelope
        :param theta: orientation of the normal to the parallel stripes of a Gabor function (radians)
        :param Lambda: wavelength of the sinusoidal factor
        :param gamma: ppatial aspect ratio
        :param psi: phase offset
        :return: gabor_kernel (np.ndarray)
        """
        # default value for psi
        if psi == 0:
            psi = math.pi * 0.5

        sigma_x = sigma
        sigma_y = sigma / gamma
        nstds = 3.

        c = math.cos(theta)
        s = math.sin(theta)

        if width > 0:
            x_max = math.floor(width / 2.)
        else:
            x_max = round(max(abs(nstds * sigma_x * s), abs(nstds * sigma_y * c)))

        if height > 0:
            y_max = math.floor(height / 2.)
        else:
            y_max = round(max(abs(nstds * sigma_x * s), abs(nstds * sigma_y * c)))

        x_min = - x_max
        y_min = - y_max
        x_max = int(x_max)
        y_max = int(y_max)
        x_min = int(x_min)
        y_min = int(y_min)

        ny = int(y_max - y_min + 1)
        nx = int(x_max - x_min + 1)
        gabor_kernel = np.zeros((ny, nx))
        scale = 1
        e_x = -0.5 / (sigma_x * sigma_x)
        e_y = -0.5 / (sigma_y * sigma_y)
        cscale = math.pi *2 / Lambda

        for y in range(y_min, y_max + 1, 1):
            for x in range(x_min, x_max + 1, 1):
                x_r = x * c + y * s
                y_r = - x * s + y * c
                gabor_kernel[y_max - y, x_max - x] = \
                    scale * math.exp(e_x * x_r * x_r + e_y * y_r * y_r) * \
                    math.cos(cscale * x_r * psi)

        return gabor_kernel


    def apply_filter_bank(self, img, Filter_Bank):
        """
        function of applying filter bank
        :param img: the input image to be processed
        :param Filter_Bank: filter bank from self.init_gabor_filter_bank()
        :return: (Features, features_dims)
        """
        Features = []
        features_dims = 0
        for i in range(len(Filter_Bank)):
            R_1 = self.filter_2D(img, Filter_Bank[i][0], 'reflect')
            R_2 = self.filter_2D(img, Filter_Bank[i][1], 'reflect')

            R = np.maximum(R_1, R_2)
            Integral_Sum = integralImage(R)
            Integral_Sq = integralImage(R * R)

            h_half = Filter_Bank[i][2] / 2.
            w_half = Filter_Bank[i][2] / 2.

            n_rows = int(math.floor((R.shape[0] + h_half - 1) / h_half))
            n_cols = int(math.floor((R.shape[1] + w_half - 1) / w_half))

            Feature = list(np.zeros(n_rows * n_cols))
            pos = 0
            for y_c in range(0, R.shape[0], int(h_half)):
                y_0 = max(0, y_c - int(h_half))
                y_1 = min(R.shape[0], y_c + int(h_half))

                for x_c in range(0, R.shape[1], int(w_half)):
                    x_0 = max(0, x_c - int(w_half))
                    x_1 = min(R.shape[1], x_c + int(w_half))

                    area = (y_1 - y_0) * (x_1 - x_0)

                    mean = Integral_Sum[y_1, x_1] - Integral_Sum[y_1, x_0] - \
                           Integral_Sum[y_0, x_1] + Integral_Sum[y_0, x_0]
                    mean = float(mean) / float(area)

                    sd = Integral_Sq[y_1, x_1] - Integral_Sq[y_1, x_0] - Integral_Sq[y_0, x_1] + Integral_Sq[y_0, x_0]
                    sd = math.sqrt(max(0.0, float(sd) / float(area) - mean * mean))

                    Feature[pos] = sd
                    pos += 1
            Features.append(Feature)
            features_dims += len(Feature)

        return (Features, features_dims)


    def filter_2D(self, img, kernel, border_type=None):
        """
        Opencv cv::filter_2D to Python filter_2D Conversion
        :param img: input image to be processed
        :param kernel: convolution kernel (or rather a correlation kernel),
                       a single-channel floating point matrix
        :param border_type: pixel extrapolation method
        :return: image result through garbor filtering.
        """
        assert isinstance(img, np.ndarray)
        assert isinstance(kernel, np.ndarray)
        i_rows, i_cols = img.shape
        k_rows, k_cols = kernel.shape

        # Force kernel dimensions to be odd values
        if k_rows % 2 == 0 or k_cols % 2 == 0:
            print('kernel dimensions must be odd')

        # Initialize output matrix
        output = np.zeros((i_rows, i_cols))

        # Get size of padding
        r_pad = int(math.floor(k_rows / 2))
        c_pad = int(math.floor(k_cols / 2))

        # Pad following OpenCV BORDER_REPLICATE
        if border_type == 'replicate':
            img_pad = np.pad(img, ((r_pad, r_pad), (c_pad, c_pad)), mode='edge')
        # Pad using rows and columns of zeros
        elif border_type == 'zeros':
            img_pad = np.pad(img, ((r_pad, r_pad), (c_pad, c_pad)), mode='constant', constant_values=(0, 0))
        # (By default) Pad following OpenCV BORDER_DEFAULT
        elif border_type == 'reflect' or border_type == None:
            if r_pad >= i_rows or c_pad >= i_cols:
                print('For reflect padding, image dimensions must be '
                      'greater than two times kernel dimensions plus one')
                return

            # Start out with 0 pad
            img_pad = np.pad(img, ((r_pad, r_pad), (c_pad, c_pad)), mode='constant', constant_values=(0, 0))

            # Perform left and right side padding
            ip_rows, ip_cols = img_pad.shape
            for r in range(i_rows):
                for c in range(c_pad):
                    img_pad[r + r_pad, c_pad - c + 1] = img[r, c + 1]
                    img_pad[r + r_pad, i_cols + c_pad + c] = img[r, i_cols - c - 2]

            # Perform top and bottom padding
            for r in range(r_pad):
                for c in range(i_cols):
                    img_pad[r_pad - r + 1, c + c_pad] = img[r + 1, c]
                    img_pad[i_rows + r_pad + r, c + c_pad] = img[i_rows - r - 2, c]

            # Perform top left corner padding
            img_pad[0:r_pad, 0:c_pad] = np.rot90(img[1:r_pad+1, 1:c_pad+1], 2)
            # Perform top right corner padding
            img_pad[0:r_pad, ip_cols-c_pad:ip_cols] = np.rot90(img[1:r_pad+1, i_cols-c_pad-1:i_cols-1], 2)
            # Perform bottom left corner padding
            img_pad[ip_rows-r_pad:ip_rows, 0:c_pad] = np.rot90(img[i_rows-r_pad-1:i_rows-1, 1:c_pad+1], 2)
            # Perform bottom right corner padding
            img_pad[ip_rows-r_pad:ip_rows, ip_cols-c_pad:ip_cols+1] =\
                np.rot90(img[i_rows-r_pad-1:i_rows-1, i_cols-c_pad-1:i_cols-1], 2)
        else:
            print('Unsupported border_type argument')
            print('Supported types: replicate, reflection, zeros')
            return

        # Perform convolution (or rather actually correlation)
        for r in range(i_rows):
            for c in range(i_cols):
                output[r, c] = np.sum(kernel * img_pad[r:r+k_rows, c:c+k_cols])

        return output


    def Get_output_BIF(self, Features, features_dims):
        """
        function of geting output BIF feature vector
        :param Features: 
        :param features_dims: 
        :return: BIF feature vector (np.ndarray, size of (features_dims, 1))
        """
        output = np.zeros((features_dims, 1))
        offset = 0
        for i in range(len(Features)):
            f_size = len(Features[i])
            output[offset:offset+f_size, 0] = Features[i]
            offset += f_size

        return output