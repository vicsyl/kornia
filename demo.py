import cv2 as cv
from kornia.constants import pi
import torch
import kornia
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from kornia.feature import get_laf_orientation, set_laf_orientation, extract_patches_from_pyramid
from patch_dominant_go import CustomizablePatchDominantGradientOrientation


def timg_load(fname, to_gray = True):
    img = cv.imread(fname)
    with torch.no_grad():
        timg = kornia.image_to_tensor(img, False).float()
        if to_gray:
            timg = kornia.color.bgr_to_grayscale(timg)
        else:
            timg = kornia.color.bgr_to_rgb(timg)
    return timg


def estimate_angle(img: torch.tensor, laf, alpha, ori_module):
    """ Estimate dominant angle directly by
    a) extracting the rotated path (rotated by alpha)
    b) directly calling ori_module.forward()
    Args:
        img: self-explanatory
        laf: self-explanatory
        alpha: self-explanatory
    Returns:
        estimated angle
    """

    orig_angle = get_laf_orientation(laf)
    laf_current = set_laf_orientation(laf, alpha + orig_angle)
    patch_rotated = extract_patches_from_pyramid(img, laf_current)[0]
    estimated_angle = ori_module(patch_rotated).reshape(-1)
    estimated_alpha = estimated_angle * 180 / pi # - orig_angle
    estimated_alpha = estimated_alpha.item()
    if estimated_alpha > 180:
        estimated_alpha = estimated_alpha - 360
    elif estimated_alpha < -180:
        estimated_alpha = estimated_alpha + 360
    return estimated_alpha


def estimate_angle_artificially_rotated(img: torch.tensor, laf, alpha, ori_module):
    """ Estimate dominant angle directly by
    a) extracting the original path (not rotated)
    b) computing the gradients via ori_module
    c) rotating the grads
    d) estimating the angle via ori_module, but based on the gradients
    Args:
        img: self-explanatory
        laf: self-explanatory
        alpha: self-explanatory
    Returns:
        estimated angle
    """
    #ori_module = CustomizablePatchDominantGradientOrientation(32)
    #orig_angle = get_laf_orientation(laf)
    #laf_current = set_laf_orientation(laf, alpha + orig_angle)
    patch = extract_patches_from_pyramid(img, laf)[0]

    # get the grads
    grads = ori_module.compute_grads(patch)
    gx: torch.Tensor = grads[:, :, 0]
    gy: torch.Tensor = grads[:, :, 1]
    norm: torch.Tensor = torch.sqrt(gx * gx + gy * gy)
    ori = torch.atan2(gy, gx + 1e-18) + 2.0 * pi

    # rotate grads by alpha
    ori = ori + alpha * pi / 180
    grads[:, :, 0] = torch.cos(ori) * norm  # gx
    grads[:, :, 1] = torch.sin(ori) * norm  # gy

    estimated_angle = ori_module(patch, grads).reshape(-1)
    estimated_alpha = estimated_angle * 180 / pi
    return estimated_alpha.item()


default_alphas = np.arange(0, 30, 1)
save_plots = True


def plot_measurements(title, measurements_labels, x_measurements=default_alphas, save=save_plots):

    plt.figure(figsize=(6, 6))
    plt.title(title)

    colors = ["y", "b", "m", "k", "g"] * 10

    for m_l, c in zip(measurements_labels, colors[:len(measurements_labels)]):
        m, l = m_l
        plt.plot(x_measurements, m, color=c, label=l)

    plt.plot(x_measurements, x_measurements, 'r--', label="y=x")
    plt.grid()
    plt.legend()
    if save:
        plt.savefig(f"./plots/{title.replace(' ', '_')}.png")
    plt.show()
    plt.close()


def imgs_lafs():
    timg = timg_load('graffiti.ppm', False) / 255.
    timg_gray = kornia.color.rgb_to_grayscale(timg)
    # A = torch.tensor([[[50., 0., 300.],
    #                    [0., 50., 200.],
    #                    [0., 0., 1.]]])
    # laf = A[:, :2, :].reshape(1, 1, 2, 3)
    from kornia.feature import SIFTFeature
    sf = SIFTFeature()
    (lafs, responses, descs) = sf(timg_gray)
    return lafs


def exp_for_laf(timg_gray, laf):

    ori_module = CustomizablePatchDominantGradientOrientation(32)
    ori_module_refined = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True})
    ori_module_refined2 = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "ref": True})
    ori_module_refined_gauss = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "gauss_weighted": True})
    ori_module_refined_circular = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "circular_kernel": True})
    ori_module_refined_circular_precise = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "circular_kernel_precise": True})
    ori_module_refined_gauss_cp = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "gauss_weighted": True, "circular_kernel_precise": True})

    # from kornia_moons.feature import visualize_LAF
    # visualize_LAF(timg, kornia.feature.scale_laf(laf, 2.0), color='yellow')

    #zero_angle = estimate_angle(timg_gray, laf, 0.0, ori_module)
    #zero_gt = estimate_angle(timg_gray, laf, 0.0, ori_module_refined)
    zero_gt = estimate_angle(timg_gray, laf, 0.0, ori_module_refined_circular_precise)
    zero_gt_grad = estimate_angle_artificially_rotated(timg_gray, laf, 0.0, ori_module_refined)

    def estimate(module):
        return [estimate_angle(timg_gray, laf, -alpha, module) - zero_gt for alpha in tqdm(default_alphas)]

    def estimate_grad_rot(module):
        return [estimate_angle_artificially_rotated(timg_gray, laf, -alpha, module) - zero_gt_grad for alpha in tqdm(default_alphas)]

    # redo to use 'estimate(module):'
    est_alphas_artificial = (estimate_grad_rot(ori_module), "Kornia grad rot. baseline")
    est_alphas_artificial_refined = (estimate_grad_rot(ori_module_refined), "grad. rot. par. refined")
    est_alphas_artificial_refined2 = (estimate_grad_rot(ori_module_refined2), "grad. rot. par. refined refined")

    # plot_measurements("Kornia - grad rotation", [est_alphas_artificial, est_alphas_artificial_refined], ["Kornia", "Kornia parabola refinement"])

    est_alphas_original = (estimate(ori_module), "Kornia baseline")
    est_alphas_original_refined = (estimate(ori_module_refined), "par. refined")
    est_alphas_original_refined_circular = (estimate(ori_module_refined_circular), "par. circular coarse")
    est_alphas_original_refined_gauss = (estimate(ori_module_refined_gauss), "par. gauss")
    est_alphas_original_refined_gauss_cp = (estimate(ori_module_refined_gauss_cp), "par. gauss circ. precise")
    est_alphas_original_refined_circular_precise = (estimate(ori_module_refined_circular_precise), "par. circ. precise")
    estimates_labels = [est_alphas_original,
                        est_alphas_original_refined,
                        est_alphas_original_refined_circular_precise,
                        est_alphas_original_refined_gauss,
                        est_alphas_original_refined_gauss_cp]
    plot_measurements(f"Kornia laf={laf}", estimates_labels)

    # estimates = [est_alphas_original, est_alphas_original_refined, est_alphas_original_refined_circular_precise, est_alphas_original_refined_gauss]
    # labels = ["Kornia baseline", "refined", "And circular refined", "gauss"]
    # estimates = [est_alphas_artificial, est_alphas_artificial_refined]
    # labels = ["grad rot", "grad rot refined"]

    plot_measurements(f"Kornia grad. rot. laf={laf}", [est_alphas_artificial, est_alphas_artificial_refined, est_alphas_artificial_refined2])

    # plt.figure("Basic plot")
    # plt.plot(default_alphas, est_alphas_original, label="Kornia original")
    # plt.plot(default_alphas, est_alphas_original_refined, label="Kornia original refined")
    # plt.plot(default_alphas, est_alphas_artificial, label="Kornia grad rotation")
    # plt.plot(default_alphas, est_alphas_artificial_refined, label="Kornia grad rotation refined")
    # plt.plot(default_alphas, default_alphas, 'r:', label="y=x")
    # plt.grid()
    # plt.legend()
    # plt.show()


def main():

    timg = timg_load('graffiti.ppm', False) / 255.
    timg_gray = kornia.color.rgb_to_grayscale(timg)
    A = torch.tensor([[[50., 0., 300.],
                       [0., 50., 200.],
                       [0., 0., 1.]]])
    laf_manual = A[:, :2, :].reshape(1, 1, 2, 3)
    exp_for_laf(timg_gray, laf_manual)

    detected_lafs = imgs_lafs()
    scale = 1.0
    for i in range(10):
        laf = detected_lafs[:, i:i + 1]
        laf[:, :, :, :2] *= scale
        exp_for_laf(timg_gray, detected_lafs[:, i:i+1])

    # ori_module = CustomizablePatchDominantGradientOrientation(32)
    # ori_module_refined = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True})
    # ori_module_refined_gauss = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "gauss_weighted": True})
    # ori_module_refined_circular = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "circular_kernel": True})
    # ori_module_refined_circular_precise = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "circular_kernel_precise": True})
    # ori_module_refined_gauss_cp = CustomizablePatchDominantGradientOrientation(32, conf={"refined": True, "gauss_weighted": True, "circular_kernel_precise": True})
    #
    # # from kornia_moons.feature import visualize_LAF
    # # visualize_LAF(timg, kornia.feature.scale_laf(laf, 2.0), color='yellow')
    #
    # #zero_angle = estimate_angle(timg_gray, laf, 0.0, ori_module)
    # #zero_gt = estimate_angle(timg_gray, laf, 0.0, ori_module_refined)
    # zero_gt = estimate_angle(timg_gray, laf, 0.0, ori_module_refined_circular_precise)
    # zero_gt_grad = estimate_angle_artificially_rotated(timg_gray, laf, 0.0, ori_module_refined)
    #
    # def estimate(module):
    #     return [estimate_angle(timg_gray, laf, -alpha, module) - zero_gt for alpha in tqdm(default_alphas)]
    #
    # def estimate_grad_rot(module):
    #     return [estimate_angle_artificially_rotated(timg_gray, laf, -alpha, module) - zero_gt_grad for alpha in tqdm(default_alphas)]
    #
    # # redo to use 'estimate(module):'
    # est_alphas_artificial = (estimate_grad_rot(ori_module), "Kornia grad rot. baseline")
    # est_alphas_artificial_refined = (estimate_grad_rot(ori_module_refined), "grad. rot. par. refined")
    #
    # # plot_measurements("Kornia - grad rotation", [est_alphas_artificial, est_alphas_artificial_refined], ["Kornia", "Kornia parabola refinement"])
    #
    # est_alphas_original = (estimate(ori_module), "Kornia baseline")
    # est_alphas_original_refined = (estimate(ori_module_refined), "par. refined")
    # est_alphas_original_refined_circular = (estimate(ori_module_refined_circular), "par. circular coarse")
    # est_alphas_original_refined_gauss = (estimate(ori_module_refined_gauss), "par. gauss")
    # est_alphas_original_refined_gauss_cp = (estimate(ori_module_refined_gauss_cp), "par. gauss circ. precise")
    # est_alphas_original_refined_circular_precise = (estimate(ori_module_refined_circular_precise), "par. circ. precise")
    # estimates_labels = [est_alphas_original,
    #                     est_alphas_original_refined,
    #                     est_alphas_original_refined_circular_precise,
    #                     est_alphas_original_refined_gauss,
    #                     est_alphas_original_refined_gauss_cp]
    # plot_measurements("Kornia", estimates_labels)
    #
    # # estimates = [est_alphas_original, est_alphas_original_refined, est_alphas_original_refined_circular_precise, est_alphas_original_refined_gauss]
    # # labels = ["Kornia baseline", "refined", "And circular refined", "gauss"]
    # # estimates = [est_alphas_artificial, est_alphas_artificial_refined]
    # # labels = ["grad rot", "grad rot refined"]
    #
    # plot_measurements("Kornia grad. rot.", [est_alphas_artificial, est_alphas_artificial_refined])
    #
    # # plt.figure("Basic plot")
    # # plt.plot(default_alphas, est_alphas_original, label="Kornia original")
    # # plt.plot(default_alphas, est_alphas_original_refined, label="Kornia original refined")
    # # plt.plot(default_alphas, est_alphas_artificial, label="Kornia grad rotation")
    # # plt.plot(default_alphas, est_alphas_artificial_refined, label="Kornia grad rotation refined")
    # # plt.plot(default_alphas, default_alphas, 'r:', label="y=x")
    # # plt.grid()
    # # plt.legend()
    # # plt.show()


if __name__ == "__main__":
    main()

# CONTINUE:
# epsilon vs Gauss
# other impls
# more patches/imgs