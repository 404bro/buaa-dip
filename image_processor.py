from PIL import Image, ImageOps, ImageEnhance, ImageChops
import cv2
import numpy as np
import numpy.fft as fft
from scipy.signal import convolve2d


def add_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    if image1.mode != "RGB":
        image1 = image1.convert("RGB")
    if image2.mode != "RGB":
        image2 = image2.convert("RGB")

    image2_resized = image2.resize(image1.size)
    return ImageChops.add(image1, image2_resized)


def add_brightness(image: Image.Image, value: int) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1 + value / 100.0)


def invert(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        rgb_image = Image.merge("RGB", (r, g, b))
        inverted_image = ImageOps.invert(rgb_image)
        r2, g2, b2 = inverted_image.split()
        return Image.merge("RGBA", (r2, g2, b2, a))
    else:
        return ImageOps.invert(image)


def translate(image: Image.Image, dx: int, dy: int) -> Image.Image:
    return image.transform(image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))


def flip(image: Image.Image, direction: str) -> Image.Image:
    if direction == "horizontal":
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == "vertical":
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    return image


def rotate(image: Image.Image, angle: int) -> Image.Image:
    return image.rotate(angle, expand=True)


def histogram_equalization(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        r, g, b, a = image.split()
        r_eq = ImageOps.equalize(r)
        g_eq = ImageOps.equalize(g)
        b_eq = ImageOps.equalize(b)
        return Image.merge("RGBA", (r_eq, g_eq, b_eq, a))
    else:
        return ImageOps.equalize(image)


def homomorphic_filter(
    image: Image.Image, a=0.5, b=1.5, cutoff=30, order=2
) -> Image.Image:
    """
    使用同态滤波增强图像。
    :param a: 控制振幅范围的下限 (gamma_l)
    :param b: 控制振幅范围的上限 (gamma_h)
    :param cutoff: 截止频率
    :param order: 巴特沃斯滤波器的阶数
    """
    img_gray = image.convert("L")
    img = np.array(img_gray, dtype=np.float64)

    # 1. 取对数
    img_log = np.log(img + 1.0)

    # 2. 傅里叶变换
    img_fft = np.fft.fft2(img_log)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # 3. 创建高通滤波器
    rows, cols = img.shape
    u, v = np.meshgrid(np.arange(cols) - cols // 2, np.arange(rows) - rows // 2)
    dist = np.sqrt(u**2 + v**2)
    H = 1 / (1 + (dist / cutoff) ** (2 * order))
    H_high = 1 - H

    # 4. 应用滤波器，调整振幅
    H_final = a + b * H_high
    img_fft_filtered = H_final * img_fft_shifted

    # 5. 傅里叶反变换
    img_ifft_shifted = np.fft.ifftshift(img_fft_filtered)
    img_ifft = np.fft.ifft2(img_ifft_shifted)

    # 6. 取指数
    img_exp = np.exp(np.real(img_ifft)) - 1.0

    # 7. 归一化并转回 uint8
    img_result = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))
    img_result = np.uint8(img_result * 255)

    return Image.fromarray(img_result)


def gamma_correction(image: Image.Image, gamma: float) -> Image.Image:
    if gamma <= 0:
        raise ValueError("Gamma值必须大于0")
    inv_gamma = 1.0 / gamma
    lut = [pow(i / 255.0, inv_gamma) * 255 for i in range(256)]
    lut = np.array(lut, dtype=np.uint8)
    return Image.fromarray(cv2.LUT(np.array(image), lut))


def laplacian_sharpen(image: Image.Image) -> Image.Image:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    img_array = np.array(image.convert("RGB"))
    sharpened_channels = []
    for i in range(3):
        channel = img_array[:, :, i]
        sharpened_channel = convolve2d(channel, kernel, mode="same", boundary="symm")
        sharpened_channels.append(np.clip(sharpened_channel, 0, 255))

    sharpened_img = np.stack(sharpened_channels, axis=-1).astype(np.uint8)
    return Image.fromarray(sharpened_img)


def edge_detection(image: Image.Image, operator: str) -> Image.Image:
    img_gray = np.array(image.convert("L"))

    if operator == "roberts":
        kx = np.array([[1, 0], [0, -1]])
        ky = np.array([[0, 1], [-1, 0]])
        grad_x = convolve2d(img_gray, kx, mode="same", boundary="symm")
        grad_y = convolve2d(img_gray, ky, mode="same", boundary="symm")
    elif operator == "prewitt":
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        grad_x = convolve2d(img_gray, kx, mode="same", boundary="symm")
        grad_y = convolve2d(img_gray, ky, mode="same", boundary="symm")
    elif operator == "sobel":
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    elif operator == "laplacian":
        lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)
        return Image.fromarray(cv2.convertScaleAbs(lap))
    else:
        return image

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = (gradient_magnitude / np.max(gradient_magnitude) * 255).astype(
        np.uint8
    )
    return Image.fromarray(gradient_magnitude)


def dilate(image: Image.Image, kernel_size=3) -> Image.Image:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_cv = np.array(image)
    dilated_img = cv2.dilate(img_cv, kernel, iterations=1)
    return Image.fromarray(dilated_img)


def erode(image: Image.Image, kernel_size=3) -> Image.Image:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_cv = np.array(image)
    eroded_img = cv2.erode(img_cv, kernel, iterations=1)
    return Image.fromarray(eroded_img)


def sift_match(image1: Image.Image, image2: Image.Image) -> Image.Image:
    if image1.mode != "RGB":
        image1 = image1.convert("RGB")
    if image2.mode != "RGB":
        image2 = image2.convert("RGB")

    img1_cv = cv2.cvtColor(np.array(image1, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    img2_cv = cv2.cvtColor(np.array(image2, dtype=np.uint8), cv2.COLOR_RGB2BGR)

    h1, w1 = img1_cv.shape[:2]
    h2, w2 = img2_cv.shape[:2]

    if h1 != h2:
        if h1 > h2:
            new_w2 = int(w2 * h1 / h2)
            img2_cv = cv2.resize(img2_cv, (new_w2, h1), interpolation=cv2.INTER_AREA)
        else:
            new_w1 = int(w1 * h2 / h1)
            img1_cv = cv2.resize(img1_cv, (new_w1, h2), interpolation=cv2.INTER_AREA)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_cv, None)
    kp2, des2 = sift.detectAndCompute(img2_cv, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    if des1 is not None and des2 is not None:
        matches = bf.knnMatch(des1, des2, k=2)
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])

    match_img = cv2.drawMatchesKnn(
        img1_cv,
        kp1,
        img2_cv,
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return Image.fromarray(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))


def fft_transform_display(image: Image.Image) -> Image.Image:

    # Step 1 & 2
    img_gray = image.convert("L")
    img_array = np.array(img_gray)
    f_transform = fft.fft2(img_array)
    f_transform_shifted = fft.fftshift(f_transform)

    # Step 3
    magnitude_spectrum = np.abs(f_transform_shifted)
    log_spectrum = np.log(1 + magnitude_spectrum)
    min_val, max_val = np.min(log_spectrum), np.max(log_spectrum)
    if max_val == min_val:
        return Image.fromarray(np.uint8(log_spectrum))

    spectrum_visual = (log_spectrum - min_val) / (max_val - min_val)
    spectrum_visual = np.uint8(spectrum_visual * 255)

    return Image.fromarray(spectrum_visual)


def fft_conjugate_rotation(image: Image.Image) -> Image.Image:
    img_gray = image.convert("L")
    img_array = np.array(img_gray, dtype=np.float64)
    rows, cols = img_array.shape

    # (a) 用 (-1)^(x+y) 乘以图像
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x, y)
    center_mult = (-1) ** (xx + yy)
    img_centered = img_array * center_mult

    # (b) 计算DFT
    f_transform = fft.fft2(img_centered)

    # (c) 取变换的复共轭
    f_conjugate = np.conj(f_transform)

    # (d) 计算傅里叶反变换
    img_ifft = fft.ifft2(f_conjugate)

    # (e) 用 (-1)^(x+y) 乘以结果的实部
    result_real = np.real(img_ifft)
    result_final = result_real * center_mult

    result_final = np.clip(result_final, 0, 255)
    return Image.fromarray(result_final.astype(np.uint8))


def reconstruct_with_fourier_descriptors(
    image: Image.Image, num_descriptors: int
) -> Image.Image:
    if num_descriptors < 2:
        num_descriptors = 2

    # 1. 寻找轮廓/边界
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise ValueError("在图像中未找到可用的轮廓。")
    contour = max(contours, key=cv2.contourArea)

    # 2. 将边界表示为复数序列 z(k) = x(k) + j*y(k)
    contour_points = contour.squeeze()
    complex_contour = contour_points[:, 0] + 1j * contour_points[:, 1]

    # 3. 计算傅里叶描述子 (1D-FFT)
    descriptors = fft.fft(complex_contour)

    # 4. 用不同项数重构：保留低频分量，去除高频分量
    truncated_descriptors = np.zeros_like(descriptors)
    half = (num_descriptors + 1) // 2
    truncated_descriptors[0:half] = descriptors[0:half]
    if num_descriptors > 1:
        truncated_descriptors[-(half - 1) :] = descriptors[-(half - 1) :]

    # 5. IFFT反变换，得到重构的边界点
    reconstructed_complex = fft.ifft(truncated_descriptors)
    reconstructed_points = np.array(
        [reconstructed_complex.real, reconstructed_complex.imag]
    ).T
    reconstructed_points = reconstructed_points.astype(np.int32)

    # 6. 在新画布上绘制重构的边界
    output_image = np.zeros_like(img_cv)
    cv2.polylines(
        output_image,
        [reconstructed_points],
        isClosed=True,
        color=(255, 255, 255),
        thickness=2,
    )

    return Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
