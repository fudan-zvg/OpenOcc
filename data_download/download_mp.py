import argparse
import os
import tempfile
import urllib.request
import time
import zipfile
import logging
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = 'http://kaldir.vc.in.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    'cameras',
    'matterport_camera_intrinsics',
    'matterport_camera_poses',
    'matterport_color_images',
    'matterport_depth_images',
    'matterport_hdr_images',
    'matterport_mesh',
    'matterport_skybox_images',
    'undistorted_camera_parameters',
    'undistorted_color_images',
    'undistorted_depth_images',
    'undistorted_normal_images',
    'house_segmentations',
    'region_segmentations',
    'image_overlap_data',
    'poisson_meshes',
    'sens'
]
TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': ['mp3d_360/data_00.zip', 'mp3d_360/data_01.zip', 'mp3d_360/data_02.zip', 'mp3d_360/data_03.zip', 'mp3d_360/data_04.zip', 'mp3d_360/data_05.zip', 'mp3d_360/data_06.zip']
}

def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def get_scans_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            scans = [line.strip() for line in file.readlines()]
        return scans
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []

def download_release(release_scans, out_dir, file_types):
    logging.info(f'Starting MP release download to {out_dir}...')
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_scan, scan_id, out_dir, file_types)
            for scan_id in release_scans
        ]
        for future in as_completed(futures):
            future.result()
    logging.info('MP release download completed.')

def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        start_time = time.time()
        with urllib.request.urlopen(url) as response, open(out_file_tmp, 'wb') as out_file_handle:
            file_size = int(response.getheader('Content-Length'))
            chunk_size = 1024
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(out_file), ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file_handle.write(chunk)
                    pbar.update(len(chunk))
        os.rename(out_file_tmp, out_file)
        end_time = time.time()
        download_time = end_time - start_time
        download_speed = file_size / download_time / 1024
        pbar.set_postfix_str(f'{download_speed:.2f} KB/s')
        logging.info(f'Download completed: {out_file}, Time taken: {download_time:.2f} seconds, Speed: {download_speed:.2f} KB/s')
        unzip_file(out_file)
    else:
        logging.info(f'File already exists, skipping download: {out_file}')
        unzip_file(out_file)

def download_scan(scan_id, out_dir, file_types):
    logging.info(f'Starting download of MP scan {scan_id} ...')
    scan_out_dir = os.path.join(out_dir, scan_id)
    if not os.path.isdir(scan_out_dir):
        os.makedirs(scan_out_dir)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(download_file, BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip', os.path.join(scan_out_dir, ft + '.zip'))
            for ft in file_types
        ]
        for future in as_completed(futures):
            future.result()
    logging.info(f'Scan download completed: {scan_id}')

def download_task_data(task_data, out_dir):
    logging.info(f'Starting download of MP task data {str(task_data)} ...')
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for task_data_id in task_data:
            if task_data_id in TASK_FILES:
                for filepart in TASK_FILES[task_data_id]:
                    url = BASE_URL + RELEASE_TASKS + '/' + filepart
                    localpath = os.path.join(out_dir, filepart)
                    localdir = os.path.dirname(localpath)
                    if not os.path.isdir(localdir):
                        os.makedirs(localdir)
                    futures.append(executor.submit(download_file, url, localpath))
        for future in as_completed(futures):
            future.result()
    logging.info(f'Task data download completed: {str(task_data)}')

def unzip_file(file_path):
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            extract_dir = os.path.dirname(os.path.dirname(file_path))  # Extract to the parent directory, i.e., scans folder
            zip_ref.extractall(extract_dir)
        logging.info(f'Extraction completed: {file_path}')
        # Delete the zip file
        try:
            os.remove(file_path)
            logging.info(f'Deleted zip file: {file_path}')
        except OSError as e:
            logging.error(f'Failed to delete zip file: {file_path}, Error: {e}')
    else:
        logging.warning(f'File is not a ZIP format, skipping extraction: {file_path}')

def main():
    parser = argparse.ArgumentParser(description=
        '''
        Download MP public dataset release.
        Example usage:
          python download_mp.py -o base_dir --scans scans.txt --type object_segmentations --task_data semantic_voxel_label_data semantic_voxel_label_models
        -o parameter is required, specifying the local directory base_dir.
        After downloading, base_dir/v1/scans will be populated with scan data, base_dir/v1/tasks will be populated with task data.
        Extract scan files from base_dir/v1/scans and task files from base_dir/v1/tasks/task_name.
        --type parameter is optional (if not specified, all data types will be downloaded).
        --scans parameter specifies the file containing the scan IDs to download.
        --task_data parameter is optional, it will download task data and model files.
        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--out_dir', default="data/Matterport3D", help='Directory to download to')
    parser.add_argument('--scans', default="md3d_train.txt", help='File containing scan IDs')
    parser.add_argument('--task_data', default=[], nargs='+', help='Task data files to download. Any of: ' + ','.join(TASK_FILES.keys()))
    parser.add_argument('--type', nargs='+', help='Specific file types to download. Any of: ' + ','.join(FILETYPES))
    parser.add_argument('--log_file', default='download_train.log', help='Path to log file')
    args = parser.parse_args()

    setup_logging(args.log_file)

    logging.info('Press any key to confirm you have agreed to the MP Terms of Service as described below:')
    logging.info(TOS_URL)
    logging.info('***')
    input('Press any key to continue, or CTRL-C to exit.')

    release_scans = get_scans_from_file(args.scans)

    release_scans = [scanid.split(" ")[0] for scanid in release_scans]
    release_scans = list(set(release_scans))

    if not release_scans:
        logging.error(f"Failed to read scan ID list, please check the file path and content: {args.scans}")
        return

    logging.info(f"A total of {len(release_scans)} room data will be downloaded this time. Do you confirm the download? (y/n)")
    confirm = input().strip().lower()
    if confirm != 'y':
        logging.info("Download cancelled.")
        return

    file_types = FILETYPES

    # Download task data
    if args.task_data:
        if set(args.task_data) & set(TASK_FILES.keys()):  # Download task data
            out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
            download_task_data(args.task_data, out_dir)
        else:
            logging.error('Error: Unrecognized task data ID: ' + str(args.task_data))
            input('Press any key to continue downloading the main dataset, or CTRL-C to exit.')

    # Download specific file types?
    if args.type:
        if not set(args.type) & set(FILETYPES):
            logging.error('Error: Invalid file type: ' + str(args.type))
            return
        file_types = args.type

    out_dir = os.path.join(args.out_dir, RELEASE)
    download_release(release_scans, out_dir, file_types)

if __name__ == "__main__":
    main()