import os
import glob
from .PhotoGenerator import PhotoGenerator
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Optional
import time
import tarfile
import shutil

class DatasetProcessor:
    """
    A class for processing and preparing datasets for 3D reconstruction training.
    
    This class handles the extraction of tar files, organization of dataset files,
    and generation of training data from stereo images and depth maps. It supports
    parallel processing of multiple datasets and keyframes.
    
    Attributes:
        base_directory (str): Path to the directory containing raw dataset files
        output_base_directory (str): Path where processed data will be saved
        verbose (bool): Whether to print detailed progress information
        num_processes (int): Number of parallel processes to use
        excluded_folders (list): List of folder names to exclude from processing
        force_regenerate (bool): Whether to regenerate files even if they exist
    """
    
    def __init__(self, base_directory: str, output_base_directory: str, 
                 verbose: bool = False, num_processes: Optional[int] = None,
                 excluded_folders: Optional[List[str]] = None,
                 extract_tar: bool = True,
                 force_regenerate: bool = False):
        """
        Initialize the DatasetProcessor
        
        Args:
            base_directory (str): Path to the directory containing raw dataset files
            output_base_directory (str): Path where processed data will be saved
            verbose (bool): Whether to print detailed progress information
            num_processes (int, optional): Number of parallel processes to use. Defaults to None.
            excluded_folders (List[str], optional): List of folder names to exclude from processing. Defaults to None.
            extract_tar (bool): Whether to extract tar.gz files found in the dataset
            force_regenerate (bool): Whether to regenerate files even if they exist
        """
        self.base_directory = base_directory
        self.output_base_directory = output_base_directory
        self.verbose = verbose
        self.num_processes = num_processes if num_processes is not None else max(1, mp.cpu_count() - 1)
        self.excluded_folders = excluded_folders or []
        self.extract_tar = extract_tar
        self.force_regenerate = force_regenerate
        
        os.makedirs(self.output_base_directory, exist_ok=True)

    def process_all_datasets(self):
        """Process all datasets in the base directory"""
        dataset_dirs = glob.glob(os.path.join(self.base_directory, "dataset_*"))
        
        dataset_dirs = [d for d in dataset_dirs 
                       if os.path.basename(d) not in self.excluded_folders]
        
        if not dataset_dirs:
            print("No valid dataset directories found.")
            return
        
        print(f"Found {len(dataset_dirs)} dataset directories to process")
        
        if self.num_processes == 1:
            for dataset_dir in dataset_dirs:
                print(f"Processing dataset: {os.path.basename(dataset_dir)}")
                self.process_single_dataset(dataset_dir)
        else:
            with mp.Pool(processes=self.num_processes) as pool:
                args = [(dataset_dir, self.output_base_directory, self.verbose, self.extract_tar, self.force_regenerate) 
                       for dataset_dir in dataset_dirs]
                
                list(tqdm(
                    pool.imap(self._process_dataset_wrapper, args),
                    total=len(dataset_dirs),
                    desc="Processing datasets",
                    unit="dataset"
                ))

    @staticmethod
    def _process_dataset_wrapper(args):
        """Wrapper function for multiprocessing"""
        dataset_dir, output_base_directory, verbose, extract_tar, force_regenerate = args
        processor = DatasetProcessor(
            base_directory=os.path.dirname(dataset_dir),
            output_base_directory=output_base_directory,
            verbose=verbose,
            extract_tar=extract_tar,
            force_regenerate=force_regenerate
        )
        processor.process_single_dataset(dataset_dir)

    def process_single_dataset(self, dataset_dir: str):
        """Process a single dataset directory"""
        dataset_name = os.path.basename(dataset_dir)
        if self.verbose:
            print(f"Processing dataset: {dataset_name}")
        
        keyframe_dirs = glob.glob(os.path.join(dataset_dir, "keyframe_*"))
        
        keyframe_dirs = [d for d in keyframe_dirs 
                        if os.path.basename(d) not in self.excluded_folders]
        
        if not keyframe_dirs:
            print(f"No valid keyframe directories found in {dataset_name}")
            return
        
        print(f"Found {len(keyframe_dirs)} keyframe directories in {dataset_name}")
        
        for keyframe_dir in tqdm(keyframe_dirs, 
                               desc=f"Processing keyframes in {dataset_name}",
                               unit="keyframe",
                               leave=False):
            keyframe_name = os.path.basename(keyframe_dir)
            
            output_dir = os.path.join(self.output_base_directory, dataset_name, keyframe_name)
            os.makedirs(output_dir, exist_ok=True)
            
            self.process_keyframe(keyframe_dir, output_dir)

    def process_keyframe(self, keyframe_dir: str, output_dir: str):
        """
        Process a single keyframe directory
        
        Args:
            keyframe_dir (str): Path to the keyframe directory
            output_dir (str): Path where processed data should be saved
        """
        start_time = time.time()
        
        point_cloud_path = os.path.join(keyframe_dir, "point_cloud.obj")
        rgb_video_path = os.path.join(keyframe_dir, "data", "rgb.mp4")
        
        if not os.path.exists(point_cloud_path):
            print(f"Warning: point_cloud.obj not found in {keyframe_dir}")
            return
        
        if not os.path.exists(rgb_video_path):
            print(f"Warning: rgb.mp4 not found in {keyframe_dir}/data")
            return
        
        try:
            if self.extract_tar:
                self._extract_tar_files(keyframe_dir)
            
            photo_generator = PhotoGenerator(
                location=os.path.join(keyframe_dir, "data"),
                save_location=output_dir,
                verbose=self.verbose,
                force_regenerate=self.force_regenerate
            )
            
            photo_generator.generate()
            
            photo_generator.createCSV(keyframe_dir)
            
            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f"Successfully processed {os.path.basename(keyframe_dir)} in {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error processing keyframe {keyframe_dir}: {e}")
            import traceback
            print(traceback.format_exc())
    
    def _extract_tar_files(self, keyframe_dir: str):
        """
        Extract tar.gz files found in the keyframe directory
        
        Args:
            keyframe_dir (str): Path to the keyframe directory
        """
        data_dir = os.path.join(keyframe_dir, "data")
        if not os.path.exists(data_dir):
            if self.verbose:
                print(f"Data directory not found in {keyframe_dir}")
            return
        
        tar_files = glob.glob(os.path.join(data_dir, "*.tar.gz"))
        
        if not tar_files:
            if self.verbose:
                print(f"No tar.gz files found in {data_dir}")
            return
        
        for tar_file in tar_files:
            try:
                extract_dir = os.path.splitext(os.path.splitext(tar_file)[0])[0]
                
                if os.path.exists(extract_dir) and os.listdir(extract_dir):
                    if self.verbose:
                        print(f"Skipping {tar_file} - already extracted to {extract_dir}")
                    
                    tar_filename = os.path.basename(tar_file)
                    
                    if "scene_points" in tar_filename:
                        target_dir = os.path.join(data_dir, "scene_points")
                        self._move_extracted_files(extract_dir, target_dir)
                    
                    elif "frame_data" in tar_filename:
                        target_dir = os.path.join(data_dir, "frame_data")
                        self._move_extracted_files(extract_dir, target_dir)
                    
                    continue
                
                if self.verbose:
                    print(f"Extracting {tar_file}...")

                os.makedirs(extract_dir, exist_ok=True)
                
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
                
                if self.verbose:
                    print(f"Successfully extracted {tar_file} to {extract_dir}")
                
                tar_filename = os.path.basename(tar_file)
                
                if "scene_points" in tar_filename:
                    target_dir = os.path.join(data_dir, "scene_points")
                    self._move_extracted_files(extract_dir, target_dir)
                
                elif "frame_data" in tar_filename:
                    target_dir = os.path.join(data_dir, "frame_data")
                    self._move_extracted_files(extract_dir, target_dir)
                
                else:
                    if self.verbose:
                        print(f"Unknown tar.gz file type: {tar_filename}")
                
            except Exception as e:
                print(f"Error extracting {tar_file}: {str(e)}")
                continue
    
    def _move_extracted_files(self, source_dir: str, target_dir: str):
        """
        Move extracted files to the target directory, handling existing files
        
        Args:
            source_dir (str): Source directory containing extracted files
            target_dir (str): Target directory where files should be moved
        """
        if os.path.exists(target_dir) and os.listdir(target_dir):
            if self.verbose:
                print(f"Skipping file move - target directory {target_dir} already exists and has content")
            return
            
        if os.path.exists(target_dir):
            if self.verbose:
                print(f"Merging contents from {source_dir} to {target_dir}")
            
            for item in os.listdir(source_dir):
                src_path = os.path.join(source_dir, item)
                dst_path = os.path.join(target_dir, item)
                
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    if os.path.exists(dst_path):
                        for root, _, files in os.walk(src_path):
                            rel_path = os.path.relpath(root, src_path)
                            target_root = os.path.join(dst_path, rel_path)
                            os.makedirs(target_root, exist_ok=True)
                            
                            for file in files:
                                src_file = os.path.join(root, file)
                                dst_file = os.path.join(target_root, file)
                                shutil.copy2(src_file, dst_file)
                    else:
                        shutil.copytree(src_path, dst_path)
        else:
            if self.verbose:
                print(f"Moving {source_dir} to {target_dir}")
            shutil.move(source_dir, target_dir)
        
        if self.verbose:
            print(f"Successfully processed files to {target_dir}")