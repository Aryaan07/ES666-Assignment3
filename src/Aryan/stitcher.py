import os
import glob
import cv2
import numpy as np
import math
import random

class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create()

    def make_panaroma_for_images_in(self, path):
        
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching')
        # loading the images
        images = [cv2.imread(img_path) for img_path in all_images]
        ref_idx = len(images) // 2  # Set reference image in the middle

        # Calculating homographies between consecutive images
        H_final_list = self.calculate_homographies(images)

        # homography chain
        H_chain_final = self.chain_homographies(H_final_list, len(images), ref_idx)
        min_xy_coord, max_xy_coord = self.calculate_image_extent(images, H_chain_final, ref_idx)
        final_img_dim = max_xy_coord - min_xy_coord
        pan_img = np.zeros((int(final_img_dim[1]), int(final_img_dim[0]), 3))  
        for i in range(len(images)):
            pan_img = self.get_panorama_image(pan_img, images[i], H_chain_final[i], min_xy_coord)
        
        return pan_img, H_chain_final

    def calculate_homographies(self, images):
        H_final_list = []
        for i in range(len(images) - 1):
            kp1, des1 = self.sift.detectAndCompute(images[i], None)
            kp2, des2 = self.sift.detectAndCompute(images[i + 1], None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Extracting matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 
            H_final_list.append(H)

        return H_final_list

    def chain_homographies(self, H_final_list, num_images, ref_idx):
        H_chain_final = []
        for i in range(num_images):
            if i < ref_idx:
                
                H_chain = np.eye(3)
                for j in range(i, ref_idx):
                    H_chain = H_final_list[j] @ H_chain  
                H_chain_final.append(H_chain)
            elif i > ref_idx:
                
                H_chain = np.eye(3)
                for j in range(i - 1, ref_idx - 1, -1):
                    H_chain = np.linalg.inv(H_final_list[j]) @ H_chain  
                H_chain_final.append(H_chain)
            else:
                H_chain_final.append(np.eye(3))  
        return H_chain_final

    def calculate_image_extent(self, images, H_chain_final, ref_idx):
        
        corners_list = [self.image_extent(images[i], H_chain_final[i] / H_chain_final[i][2, 2]) for i in range(len(images))]
        min_xy_coord = np.amin(np.amin(corners_list, 2), 0)
        max_xy_coord = np.amax(np.amax(corners_list, 2), 0)
        return min_xy_coord, max_xy_coord

    def image_extent(self, img, H):
        
        img_corners = np.array([
            [0, 0, 1],
            [0, img.shape[1], 1],
            [img.shape[0], 0, 1],
            [img.shape[0], img.shape[1], 1]]).T
        img_corners_range = H @ img_corners
        img_corners_range /= img_corners_range[-1, :]
        return img_corners_range[:2, :]

    def get_panorama_image(self, range_img, domain_img, H, offsetXY):
        H_inv = np.linalg.inv(H)
        for i in range(range_img.shape[0]):  
            for j in range(range_img.shape[1]):  
                X_domain = np.array([j + offsetXY[0], i + offsetXY[1], 1])
                X_range = H_inv @ X_domain
                X_range /= X_range[-1]  

                if (0 < X_range[0] < domain_img.shape[1] - 1 and
                        0 < X_range[1] < domain_img.shape[0] - 1):
                    range_img[i, j] = self.bilinear_interp_for_pixel_value(domain_img, X_range)
        return range_img

    def bilinear_interp_for_pixel_value(self, img, pt):
        x, y = pt[0], pt[1]
        x_floor, y_floor = math.floor(x), math.floor(y)
        x_diff, y_diff = x - x_floor, y - y_floor
        pt_imin1_jmin1 = img[y_floor, x_floor]
        pt_iplus1_jmin1 = img[y_floor, x_floor + 1]
        pt_imin1_jplus1 = img[y_floor + 1, x_floor]
        pt_iplus1_jplus1 = img[y_floor + 1, x_floor + 1]
        pt_imin1_jmin1_wt = 1 / math.sqrt(x_diff**2 + y_diff**2 + 1e-10)
        pt_iplus1_jmin1_wt = 1 / math.sqrt((1 - x_diff)**2 + y_diff**2 + 1e-10)
        pt_imin1_jplus1_wt = 1 / math.sqrt(x_diff**2 + (1 - y_diff)**2 + 1e-10)
        pt_iplus1_jplus1_wt = 1 / math.sqrt((1 - x_diff)**2 + (1 - y_diff)**2 + 1e-10)

        # Interpolated value
        result_num = (pt_imin1_jmin1 * pt_imin1_jmin1_wt +
                      pt_iplus1_jmin1 * pt_iplus1_jmin1_wt +
                      pt_imin1_jplus1 * pt_imin1_jplus1_wt +
                      pt_iplus1_jplus1 * pt_iplus1_jplus1_wt)
        result_denom = pt_imin1_jmin1_wt + pt_iplus1_jmin1_wt + pt_imin1_jplus1_wt + pt_iplus1_jplus1_wt
        return result_num / result_denom

