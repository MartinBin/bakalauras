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
    def __init__(self, base_directory: str, output_base_directory: str, 
                 verbose: bool = False, num_processes: Optional[int] = None,
                 excluded_folders: Optional[List[str]] = None,
                 extract_tar: bool = True):
        """
        Initialize the DatasetProcessor
        
        Args:
            base_directory (str): Root directory containing dataset_xx folders
            output_base_directory (str): Root directory where processed data will be saved
            verbose (bool): Whether to print detailed progress information
            num_processes (int, optional): Number of processes to use for parallel processing.
                                        If None, uses CPU count - 1
            excluded_folders (List[str], optional): List of folder names to exclude from processing
            extract_tar (bool): Whether to extract tar.gz files found in the dataset
        """
        self.base_directory = base_directory
        self.output_base_directory = output_base_directory
        self.verbose = verbose
        self.num_processes = num_processes if num_processes is not None else max(1, mp.cpu_count() - 1)
        self.excluded_folders = excluded_folders or []
        self.extract_tar = extract_tar
        
        # Create output base directory if it doesn't exist
        os.makedirs(self.output_base_directory, exist_ok=True)

    def process_all_datasets(self):
        """Process all datasets in the base directory"""
        # Find all dataset directories
        dataset_dirs = glob.glob(os.path.join(self.base_directory, "dataset_*"))
        
        # Filter out excluded folders
        dataset_dirs = [d for d in dataset_dirs 
                       if os.path.basename(d) not in self.excluded_folders]
        
        if not dataset_dirs:
            print("No valid dataset directories found.")
            return
        
        # Create a pool of workers
        with mp.Pool(processes=self.num_processes) as pool:
            # Prepare arguments for each dataset
            args = [(dataset_dir, self.output_base_directory, self.verbose, self.extract_tar) 
                   for dataset_dir in dataset_dirs]
            
            # Process datasets in parallel with progress bar
            list(tqdm(
                pool.imap(self._process_dataset_wrapper, args),
                total=len(dataset_dirs),
                desc="Processing datasets",
                unit="dataset"
            ))

    @staticmethod
    def _process_dataset_wrapper(args):
        """Wrapper function for multiprocessing"""
        dataset_dir, output_base_directory, verbose, extract_tar = args
        processor = DatasetProcessor(
            base_directory=os.path.dirname(dataset_dir),
            output_base_directory=output_base_directory,
            verbose=verbose,
            extract_tar=extract_tar
        )
        processor.process_single_dataset(dataset_dir)

    def process_single_dataset(self, dataset_dir: str):
        """Process a single dataset directory"""
        dataset_name = os.path.basename(dataset_dir)
        if self.verbose:
            print(f"Processing dataset: {dataset_name}")
        
        # Find all keyframe directories in the dataset
        keyframe_dirs = glob.glob(os.path.join(dataset_dir, "keyframe_*"))
        
        # Filter out excluded folders
        keyframe_dirs = [d for d in keyframe_dirs 
                        if os.path.basename(d) not in self.excluded_folders]
        
        if not keyframe_dirs:
            print(f"No valid keyframe directories found in {dataset_name}")
            return
        
        # Process keyframes with progress bar
        for keyframe_dir in tqdm(keyframe_dirs, 
                               desc=f"Processing keyframes in {dataset_name}",
                               unit="keyframe",
                               leave=False):
            keyframe_name = os.path.basename(keyframe_dir)
            
            # Create output directory for this keyframe
            output_dir = os.path.join(self.output_base_directory, dataset_name, keyframe_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the keyframe
            self.process_keyframe(keyframe_dir, output_dir)

    def process_keyframe(self, keyframe_dir: str, output_dir: str):
        """
        Process a single keyframe directory
        
        Args:
            keyframe_dir (str): Path to the keyframe directory
            output_dir (str): Path where processed data should be saved
        """
        start_time = time.time()
        
        # Check if required files exist
        point_cloud_path = os.path.join(keyframe_dir, "point_cloud.obj")
        rgb_video_path = os.path.join(keyframe_dir, "data", "rgb.mp4")
        
        if not os.path.exists(point_cloud_path):
            print(f"Warning: point_cloud.obj not found in {keyframe_dir}")
            return
        
        if not os.path.exists(rgb_video_path):
            print(f"Warning: rgb.mp4 not found in {keyframe_dir}/data")
            return
        
        try:
            # Extract tar.gz files if needed
            if self.extract_tar:
                self._extract_tar_files(keyframe_dir)
            
            # Initialize PhotoGenerator for this keyframe
            photo_generator = PhotoGenerator(
                location=os.path.join(keyframe_dir, "data"),
                save_location=output_dir,
                verbose=self.verbose
            )
            
            # Generate photos from video
            photo_generator.generate()
            
            # Create CSV with frame data
            photo_generator.createCSV(keyframe_dir)
            
            if self.verbose:
                elapsed_time = time.time() - start_time
                print(f"Successfully processed {os.path.basename(keyframe_dir)} in {elapsed_time:.2f} seconds")
                
        except Exception as e:
            print(f"Error processing {keyframe_dir}: {str(e)}")
    
    def _extract_tar_files(self, keyframe_dir: str):
        """
        Extract tar.gz files found in the keyframe directory
        
        Args:
            keyframe_dir (str): Path to the keyframe directory
        """
        # Look for tar.gz files in the data directory
        data_dir = os.path.join(keyframe_dir, "data")
        if not os.path.exists(data_dir):
            if self.verbose:
                print(f"Data directory not found in {keyframe_dir}")
            return
        
        # Find all tar.gz files
        tar_files = glob.glob(os.path.join(data_dir, "*.tar.gz"))
        
        if not tar_files:
            if self.verbose:
                print(f"No tar.gz files found in {data_dir}")
            return
        
        # Extract each tar.gz file
        for tar_file in tar_files:
            try:
                # Create extraction directory path
                extract_dir = os.path.splitext(os.path.splitext(tar_file)[0])[0]
                
                # Check if already extracted
                if os.path.exists(extract_dir) and os.listdir(extract_dir):
                    if self.verbose:
                        print(f"Skipping {tar_file} - already extracted to {extract_dir}")
                    
                    # Still process the extracted files even if we skipped extraction
                    tar_filename = os.path.basename(tar_file)
                    
                    # Handle scene_points tar.gz files
                    if "scene_points" in tar_filename:
                        target_dir = os.path.join(data_dir, "scene_points")
                        self._move_extracted_files(extract_dir, target_dir)
                    
                    # Handle frame_data tar.gz files
                    elif "frame_data" in tar_filename:
                        target_dir = os.path.join(data_dir, "frame_data")
                        self._move_extracted_files(extract_dir, target_dir)
                    
                    continue
                
                if self.verbose:
                    print(f"Extracting {tar_file}...")
                
                # Create extraction directory
                os.makedirs(extract_dir, exist_ok=True)
                
                # Extract the tar.gz file
                with tarfile.open(tar_file, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
                
                if self.verbose:
                    print(f"Successfully extracted {tar_file} to {extract_dir}")
                
                # Handle different types of tar.gz files
                tar_filename = os.path.basename(tar_file)
                
                # Handle scene_points tar.gz files
                if "scene_points" in tar_filename:
                    target_dir = os.path.join(data_dir, "scene_points")
                    self._move_extracted_files(extract_dir, target_dir)
                
                # Handle frame_data tar.gz files
                elif "frame_data" in tar_filename:
                    target_dir = os.path.join(data_dir, "frame_data")
                    self._move_extracted_files(extract_dir, target_dir)
                
                # Handle any other tar.gz files
                else:
                    if self.verbose:
                        print(f"Unknown tar.gz file type: {tar_filename}")
                    # Keep the extracted files in their original location
                
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
        if os.path.exists(target_dir):
            # If the directory already exists, merge the contents
            if self.verbose:
                print(f"Merging contents from {source_dir} to {target_dir}")
            
            for item in os.listdir(source_dir):
                src_path = os.path.join(source_dir, item)
                dst_path = os.path.join(target_dir, item)
                
                if os.path.isfile(src_path):
                    # For files, copy and overwrite if necessary
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    # For directories, merge contents
                    if os.path.exists(dst_path):
                        # If target directory exists, merge contents
                        for root, _, files in os.walk(src_path):
                            rel_path = os.path.relpath(root, src_path)
                            target_root = os.path.join(dst_path, rel_path)
                            os.makedirs(target_root, exist_ok=True)
                            
                            for file in files:
                                src_file = os.path.join(root, file)
                                dst_file = os.path.join(target_root, file)
                                shutil.copy2(src_file, dst_file)
                    else:
                        # If target directory doesn't exist, copy the entire directory
                        shutil.copytree(src_path, dst_path)
        else:
            # If the directory doesn't exist, move the entire directory
            if self.verbose:
                print(f"Moving {source_dir} to {target_dir}")
            shutil.move(source_dir, target_dir)
        
        if self.verbose:
            print(f"Successfully processed files to {target_dir}")