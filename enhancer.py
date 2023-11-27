import cv2
# image = cv2.imread('test2.jpg')

# img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
# # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
# # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img_output = cv2.equalizeHist(image)
# cv2.imshow('image', img_output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


class Enhancer():
    def __init__(self):
        self.um_kernel = 3
        self.um_k = 6
        self.um_gamma = 5
        self.he_clip_limit = 1.0
        self.he_tile_grid_size = 1.0
        self.mb_kernel = 3

    def enhance(self, image, enhancements):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # img_output = cv2.equalizeHist(image)
        # img_output = cv2.cvtColor(img_output, cv2.COLOR_GRAY2RGB)

        # b, g, r = cv2.split(image)
        # b = cv2.equalizeHist(b)
        # g = cv2.equalizeHist(g)
        # r = cv2.equalizeHist(r)
        # out = cv2.merge((b, g, r))
        # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        # img_output = clahe.apply(image)
        # img_output = cv2.cvtColor(img_output, cv2.COLOR_GRAY2RGB)

        img_output = image
        for enhancement in enhancements:
            if enhancement == "blur":
                img_output = self.unsharpMask(image=img_output,
                                              kernel_size=self.um_kernel, k=self.um_k, gamma=self.um_gamma)
            if enhancement == "noise":
                img_output = self.median_blur(
                    image=img_output, kernel_size=self.mb_kernel)
            if enhancement == "contrast":
                img_output = self.histogram_eq(
                    image=img_output, clip_limit=self.he_clip_limit, tile_grid_size=self.he_tile_grid_size)

        return img_output

    def unsharpMask(self, image, kernel_size, k, gamma):
        # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_image = image
        kernel = (kernel_size, kernel_size)
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(gray_image, kernel, gamma)

        # Subtract the blurred image from the original
        high_frequency_image = cv2.subtract(gray_image, blurred_image)
        # Add the high-frequency image to the original
        # set 0 for gamma

        sharpened_image = cv2.addWeighted(
            gray_image, 1, high_frequency_image, k, 0)

        # sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2RGB)
        return sharpened_image

    def histogram_eq(self, image, clip_limit, tile_grid_size):
        tile_grid = (tile_grid_size, tile_grid_size)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid)
        img_output = clahe.apply(image)
        img_output = cv2.cvtColor(img_output, cv2.COLOR_GRAY2RGB)
        # img_output = image
        return img_output

    def median_blur(self, image, kernel_size):
        img_output = cv2.medianBlur(image, kernel_size)
        return img_output
