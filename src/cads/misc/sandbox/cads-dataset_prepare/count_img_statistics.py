import os
import argparse
import nibabel as nib
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Count image statistics for a dataset')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., 0008_ctorg)')
    return parser.parse_args()


def main():
    folder_img = '/mnt/hddb/murong/CADS-dataset'  #TODO:
    output_path = '/mnt/hddb/murong/push'  #TODO:
    args = parse_args()

    dataset = args.dataset
    img_dir = os.path.join(folder_img, dataset, 'images')
    img_fns = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])

    data = []
    spacing_x, spacing_y, spacing_z = [], [], []
    resolution_x, resolution_y, resolution_z = [], [], []

    for img_fn in img_fns:
        img_path = os.path.join(img_dir, img_fn)
        img = nib.load(img_path)
        header = img.header

        spacing = header.get_zooms()  # (x, y, z)
        resolution = img.shape  # (x, y, z)

        spacing_x.append(spacing[0])
        spacing_y.append(spacing[1])
        spacing_z.append(spacing[2])
        resolution_x.append(resolution[0])
        resolution_y.append(resolution[1])
        resolution_z.append(resolution[2])

        data.append([img_fn, spacing[0], spacing[1], spacing[2], resolution[0], resolution[1], resolution[2]])

    avg_spacing_x = sum(spacing_x) / len(spacing_x)
    avg_spacing_y = sum(spacing_y) / len(spacing_y)
    avg_spacing_z = sum(spacing_z) / len(spacing_z)
    avg_resolution_x = sum(resolution_x) / len(resolution_x)
    avg_resolution_y = sum(resolution_y) / len(resolution_y)
    avg_resolution_z = sum(resolution_z) / len(resolution_z)

    data.append(['Average', avg_spacing_x, avg_spacing_y, avg_spacing_z, avg_resolution_x, avg_resolution_y, avg_resolution_z])

    df = pd.DataFrame(data, columns=['Image ID', 'Spacing X', 'Spacing Y', 'Spacing Z', 'Resolution X', 'Resolution Y', 'Resolution Z'])
    csv_file = os.path.join(output_path, f'{dataset}.csv')
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    main()