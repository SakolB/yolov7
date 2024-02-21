import argparse
import subprocess
from pathlib import Path
import cv2

def run_detect(weights, source, img_size):
    command = ['python', 'detect.py', '--weights', weights, '--source', source, '--img-size', str(img_size), '--save-txt']
    subprocess.run(command)

def find_latest_exp_directory(base_directory):
    exp_directories = list(base_directory.glob('exp*'))
    if not exp_directories:
        return None
    latest_directory = max(exp_directories, key=lambda dir: dir.stat().st_ctime)
    return latest_directory

def find_latest_txt_file(exp_directory):
    txt_files = list(exp_directory.glob('labels/*.txt'))
    if not txt_files:
        return None
    latest_file = max(txt_files, key=lambda file: file.stat().st_ctime)
    return latest_file

def cut_and_save_chunks(latest_file, source_image_path, save_directory):
    img = cv2.imread(source_image_path)
    img_height, img_width = img.shape[:2]

    with open(latest_file, 'r') as file:
        for line_num, line in enumerate(file, 1):
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id == 74:
                x_center, y_center, width, height = map(float, parts[1:5])
                X2 = int((x_center + width / 2) * img_width)
                X1 = int((x_center - width / 2) * img_width)
                Y1 = int((y_center - height / 2) * img_height)
                Y2 = int((y_center + height / 2) * img_height)
                chunk = img[Y1:Y2, X1:X2]
                chunk_filename = f"{save_directory}/chunk_{class_id}_{line_num}.jpg"
                cv2.imwrite(chunk_filename, chunk)
                print(f"Saved chunk: {chunk_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process detect.py output and cut image chunks.')
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('--source', type=str, required=True, help='Source input image path')
    parser.add_argument('--img-size', type=int, required=True, help='Image size')

    args = parser.parse_args()

    # Run detect.py with the provided arguments
    run_detect(args.weights, args.source, args.img_size)

    base_directory = Path('runs/detect')
    latest_exp_directory = find_latest_exp_directory(base_directory)
    if latest_exp_directory:
        latest_txt_file = find_latest_txt_file(latest_exp_directory)
        if latest_txt_file:
            save_directory = latest_txt_file.parent
            cut_and_save_chunks(latest_txt_file, args.source, save_directory)
        else:
            print("No .txt files found in the latest directory.")
    else:
        print("No 'exp*' directories found.")
