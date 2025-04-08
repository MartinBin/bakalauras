import cv2
import os
from glob import glob
import numpy as np
import imageio
import csv

class PhotoGenerator:
    def __init__(self,location,save_location,verbose=False):
        self.location = location
        self.save_location = save_location
        self.verbose = verbose

    def createCSV(self,path_to_point_cloud):
        csv_file_path = os.path.join(self.save_location, "frame_data.csv")
        if os.path.exists(csv_file_path):
            print("CSV file already exists")
            return

        with open(csv_file_path, mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame", "Left_RGB", "Right_RGB", "Left_Depth", "Right_Depth","Frame_Data","Point_Cloud"])

            obj_file = os.path.join(path_to_point_cloud, 'point_cloud.obj')
            if not os.path.isfile(obj_file):
                return None

            images = sorted([f for f in os.listdir(self.save_location) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            depths = sorted([f for f in os.listdir(self.save_location) if f.lower().endswith('.tiff')])
            frame = sorted([os.path.join(os.path.join(path_to_point_cloud, 'data/frame_data/'), f)
                          for f in os.listdir(os.path.join(path_to_point_cloud, 'data/frame_data/'))
                          if f.lower().endswith('.json')])

            if len(images) % 2 != 0:
               print(f"Warning: odd number of images in {self.save_location}")
               images = images[:-1]

            left_images = sorted([img for img in images if img.startswith("left_frame")])
            right_images = sorted([img for img in images if img.startswith("right_frame")])

            if len(depths) % 2 != 0:
                print(f"Warning: odd number of depths in {self.save_location}")
                depths = depths[:-1]

            left_depths = sorted([depth for depth in depths if depth.startswith("left_depth")])
            right_depths = sorted([depth for depth in depths if depth.startswith("right_depth")])

            for i, left_img in enumerate(left_images):
                csv_writer.writerow([i, os.path.join(self.save_location,left_img),
                                     os.path.join(self.save_location,right_images[i]),
                                     os.path.join(self.save_location,left_depths[i]),
                                     os.path.join(self.save_location,right_depths[i]),
                                     frame[i],obj_file])

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
                
                print(f"Processing depth file: {depth_path}, shape: {depth_image.shape}, dtype: {depth_image.dtype}")
                
                # Get original shape and split without changing format
                depth_height, depth_width, _ = depth_image.shape

                left_depth = depth_image[:depth_height // 2]
                right_depth = depth_image[depth_height // 2:]

                left_depth_name = f'{self.save_location}/left_depth{currentframe}.tiff'
                right_depth_name = f'{self.save_location}/right_depth{currentframe}.tiff'

                print('Creating...' + left_depth_name)
                # Preserve the original data type instead of forcing uint16
                imageio.imwrite(left_depth_name, left_depth)

                print('Creating...' + right_depth_name)
                imageio.imwrite(right_depth_name, right_depth)
            else:
                print(f"Warning: No depth file for frame {currentframe}. Skipping depth processing.")

            currentframe += 1

        cam.release()
        cv2.destroyAllWindows()