import cv2
import os
from glob import glob
import numpy as np
import imageio

class PhotoGenerator:
    def __init__(self,location,save_location):
        self.location = location
        self.save_location = save_location

    def generate(self):
        cam = cv2.VideoCapture(f"{self.location}/rgb.mp4")

        try:
            if not os.path.exists(self.save_location):
                os.makedirs(self.save_location)

            if os.listdir(self.save_location):
                print("Photos are already generated from video")
                return

        except OSError:
            print("Error: Creating directory. " + 'data')

        currentframe = 0

        depth_files = sorted(glob(os.path.join(self.location, "scene_points/scene_points*.tiff")))

        while(True):
            ret, frame = cam.read()
            if not ret:
                break

            height, width, _ = frame.shape
            top_half = frame[:height // 2, :]
            bottom_half = frame[height // 2:, :]

            left_name = os.path.join(self.save_location, f'left_frame{currentframe}.jpg')
            right_name = os.path.join(self.save_location, f'right_frame{currentframe}.jpg')

            print('Creating...' + left_name)
            cv2.imwrite(left_name, top_half)

            print('Creating...' + right_name)
            cv2.imwrite(right_name, bottom_half)

            if currentframe < len(depth_files):
                depth_path = depth_files[currentframe]
                depth_image = imageio.imread(depth_path)

                print(f"Processing depth file: {depth_path}")

                if len(depth_image.shape) == 3:
                    depth_image = depth_image[:, :, 0]

                depth_height, depth_width = depth_image.shape
                if depth_height != height or depth_width != width:
                    print(f"Resizing depth image from {depth_image.shape} to match RGB frame size {frame.shape}.")
                    depth_image_resized = cv2.resize(depth_image, (width, height), interpolation=cv2.INTER_NEAREST)
                else:
                    depth_image_resized = depth_image

                left_depth = depth_image_resized[:height // 2, :]
                right_depth = depth_image_resized[height // 2:, :]

                left_depth_name = f'{self.save_location}/left_depth{currentframe}.tiff'
                right_depth_name = f'{self.save_location}/right_depth{currentframe}.tiff'

                print('Creating...' + left_depth_name)
                imageio.imwrite(left_depth_name, left_depth.astype(np.uint16))  # Save as 16-bit TIFF

                print('Creating...' + right_depth_name)
                imageio.imwrite(right_depth_name, right_depth.astype(np.uint16))  # Save as 16-bit TIFF
            else:
                print(f"Warning: No depth file for frame {currentframe}. Skipping depth processing.")

            currentframe += 1

        cam.release()
        cv2.destroyAllWindows()