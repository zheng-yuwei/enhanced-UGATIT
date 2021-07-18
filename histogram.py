# -*- coding: utf-8 -*-
"""
直方图匹配，将一张图片的直方图匹配到目标图上，使两张图的视觉感觉接近
ref https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/blob/95ba2834a358fa243665c86407b220e4e78854fe/Face_Detection/align_warp_back_multiple_dlib.py
"""
import cv2
import numpy as np


def match_histograms(src_image, ref_image, rate=1.0, image_type='HWC'):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :param rate: histograms shift ratio
    :param image_type: HWC or CHW
    :return: image_after_matching
    :rtype: image (array)
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    if image_type == 'HWC':
        src_b, src_g, src_r = cv2.split(src_image)
        ref_b, ref_g, ref_r = cv2.split(ref_image)
    elif image_type == 'CHW':
        src_b, src_g, src_r = src_image[0], src_image[1], src_image[2]
        ref_b, ref_g, ref_r = ref_image[0], ref_image[1], ref_image[2]
    else:
        raise ValueError(f'image_type only HWC or CHW, no: {image_type}')

    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    if rate < 1.0:
        ref_cdf_blue = src_cdf_blue * (1.0 - rate) + ref_cdf_blue * rate
        ref_cdf_green = src_cdf_green * (1.0 - rate) + ref_cdf_green * rate
        ref_cdf_red = src_cdf_red * (1.0 - rate) + ref_cdf_red * rate

    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # Put the image back together
    if image_type == 'HWC':
        image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    elif image_type == 'CHW':
        image_after_matching = np.array([blue_after_transform, green_after_transform, red_after_transform])
    else:
        raise ValueError(f'image_type only HWC or CHW, no: {image_type}')

    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching


def calculate_cdf(histogram: np.ndarray) -> np.ndarray:
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return: normalized_cdf: The normalized cumulative distribution function
    :rtype: array
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()

    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf


def calculate_lookup(src_cdf: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table


if __name__ == '__main__':
    import numpy as np

    kid_src = cv2.imread('dataset/kid_src.png')
    man_src = cv2.imread('dataset/man_src.png')

    # 均衡 1
    kid_match_bgr_10 = match_histograms(kid_src, man_src, rate=1)
    kid_match_hsv_10 = match_histograms(cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL),
                                        cv2.cvtColor(man_src, cv2.COLOR_BGR2HSV_FULL), rate=1)
    kid_match_hls_10 = match_histograms(cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL),
                                        cv2.cvtColor(man_src, cv2.COLOR_BGR2HLS_FULL), rate=1)
    # 均衡0.5
    kid_match_bgr_05 = match_histograms(kid_src, man_src, rate=0.5)
    kid_match_hsv_05 = match_histograms(cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL),
                                        cv2.cvtColor(man_src, cv2.COLOR_BGR2HSV_FULL), rate=0.5)
    kid_match_hls_05 = match_histograms(cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL),
                                        cv2.cvtColor(man_src, cv2.COLOR_BGR2HLS_FULL), rate=0.5)

    result = np.concatenate([np.concatenate([kid_src, man_src, man_src], axis=1),
                             np.concatenate([kid_match_bgr_10, cv2.cvtColor(kid_match_hsv_10, cv2.COLOR_HSV2BGR_FULL),
                                             cv2.cvtColor(kid_match_hls_10, cv2.COLOR_HLS2BGR_FULL)],
                                            axis=1),
                             np.concatenate([kid_match_bgr_05, cv2.cvtColor(kid_match_hsv_05, cv2.COLOR_HSV2BGR_FULL),
                                             cv2.cvtColor(kid_match_hls_05, cv2.COLOR_HLS2BGR_FULL)],
                                            axis=1)],
                            axis=0)
    cv2.imwrite('all.png', result)
    cv2.imshow('match_hsv', result)
    cv2.waitKey(0)

    # 只均衡hsv某一通道
    kid_hsv_src_10 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL)
    kid_hsv_src_05 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL)
    kid_hsv_src_10[..., 0] = kid_match_hsv_10[..., 0]
    kid_hsv_src_05[..., 0] = kid_match_hsv_05[..., 0]
    kid_hls_src_10 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL)
    kid_hls_src_05 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL)
    kid_hls_src_10[..., 0] = kid_match_hls_10[..., 0]
    kid_hls_src_05[..., 0] = kid_match_hls_05[..., 0]
    kid_match_h = np.concatenate([cv2.cvtColor(kid_hsv_src_10, cv2.COLOR_HSV2BGR_FULL),
                                  cv2.cvtColor(kid_hsv_src_05, cv2.COLOR_HSV2BGR_FULL),
                                  cv2.cvtColor(kid_hls_src_10, cv2.COLOR_HLS2BGR_FULL),
                                  cv2.cvtColor(kid_hls_src_05, cv2.COLOR_HLS2BGR_FULL)], axis=1)

    kid_hsv_src_10 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL)
    kid_hsv_src_05 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL)
    kid_hsv_src_10[..., 1] = kid_match_hsv_10[..., 1]
    kid_hsv_src_05[..., 1] = kid_match_hsv_05[..., 1]
    kid_hls_src_10 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL)
    kid_hls_src_05 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL)
    kid_hls_src_10[..., 2] = kid_match_hls_10[..., 2]
    kid_hls_src_05[..., 2] = kid_match_hls_05[..., 2]
    kid_match_s = np.concatenate([cv2.cvtColor(kid_hsv_src_10, cv2.COLOR_HSV2BGR_FULL),
                                  cv2.cvtColor(kid_hsv_src_05, cv2.COLOR_HSV2BGR_FULL),
                                  cv2.cvtColor(kid_hls_src_10, cv2.COLOR_HLS2BGR_FULL),
                                  cv2.cvtColor(kid_hls_src_05, cv2.COLOR_HLS2BGR_FULL)], axis=1)

    kid_hsv_src_10 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL)
    kid_hsv_src_05 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HSV_FULL)
    kid_hsv_src_10[..., 2] = kid_match_hsv_10[..., 2]
    kid_hsv_src_05[..., 2] = kid_match_hsv_05[..., 2]
    kid_hls_src_10 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL)
    kid_hls_src_05 = cv2.cvtColor(kid_src, cv2.COLOR_BGR2HLS_FULL)
    kid_hls_src_10[..., 1] = kid_match_hls_10[..., 1]
    kid_hls_src_05[..., 1] = kid_match_hls_05[..., 1]
    kid_match_v = np.concatenate([cv2.cvtColor(kid_hsv_src_10, cv2.COLOR_HSV2BGR_FULL),
                                  cv2.cvtColor(kid_hsv_src_05, cv2.COLOR_HSV2BGR_FULL),
                                  cv2.cvtColor(kid_hls_src_10, cv2.COLOR_HLS2BGR_FULL),
                                  cv2.cvtColor(kid_hls_src_05, cv2.COLOR_HLS2BGR_FULL)], axis=1)

    result = np.concatenate([np.concatenate([kid_src, man_src, kid_src, man_src], axis=1),
                             kid_match_h, kid_match_s, kid_match_v], axis=0)

    cv2.imwrite('one.png', result)
    cv2.imshow('match_hsv', result)
    cv2.waitKey(0)
