import os
import json
import glob


def select_imgs(folder, split):
    """
    This function scans a folder for JSON files, reads the metadata to determine the split ('train', 'val', or 'test'), and selects corresponding image files if they exist.
    """
    json_files = glob.glob(folder + '/*.json')
    json_files.sort()

    images = []
    for jsonfile in json_files:
        file = open(jsonfile)
        metadata = json.load(file)
        file.close()
        if metadata['train'] == split:
            base = os.path.basename(jsonfile).split(
                '/')[-1].split('.json')[-2][:-5]
            imagefile = os.path.join(folder, base + '_0000.nii.gz')
            if os.path.exists(imagefile):
                images.append(imagefile)

    print(f'Number of images: {len(images)}')
    return images
