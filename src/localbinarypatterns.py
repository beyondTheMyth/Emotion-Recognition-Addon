from skimage import feature
import numpy as np


class LocalBinaryPatterns:

    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp_image = feature.local_binary_pattern(image, self.num_points, self.radius, method="uniform")

        (h, w) = image.shape[:2]
        cells = 8
        cell_size_y = int(h / cells)
        cell_size_x = int(w / cells)

        hist = []
        hist_array = []

        for y in range(0, h, cell_size_y):
            row = []
            cell_lbp = []
            for x in range(0, w, cell_size_x):
                cell_lbp = lbp_image[y:y + cell_size_y, x: x + cell_size_x]
                cell_LBP = cell_lbp.ravel()
                (cell_hist, _) = np.histogram(cell_LBP, bins=np.arange(0, self.num_points + 3),
                                              range=(0, self.num_points + 2))
                cell_hist = cell_hist.astype("float")
                cell_hist /= (cell_hist.sum() + eps)
                hist_array.append(cell_hist)

        hist = [item for sublist in hist_array for item in sublist]
        hist = np.asarray(hist)

        return hist
