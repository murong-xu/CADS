
from matplotlib.colors import Normalize, ListedColormap
import nibabel as nib
import numpy as np
import nibabel.orientations as nio
from scipy.ndimage.measurements import center_of_mass
import json
import matplotlib.pyplot as plt
import random
from cads.utils.colormap import get_colors

colors = get_colors()
cm_itk = ListedColormap(colors)
cm_itk.set_bad(color='w', alpha=0)


def reorient_centroids_to(ctd_list, img, decimals=1, verb=False):
    # reorient centroids to image orientation
    # todo: reorient to given axcodes (careful if img ornt != ctd ornt)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present")
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Centroids reoriented from",
              nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list


def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False):
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print("[*] Image reoriented from",
              nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg


def calc_centroids(msk, decimals=1, world=False):
    # Centroids are in voxel coordinates!
    # world=True: centroids are in world coordinates
    msk_data = np.asanyarray(msk.dataobj, dtype=msk.dataobj.dtype)
    axc = nio.aff2axcodes(msk.affine)
    ctd_list = [axc]
    verts = np.unique(msk_data)[1:]  # exclude background in seg nifti file
    verts = verts[~np.isnan(verts)]  # remove NaN values
    for i in verts:
        msk_temp = np.zeros(msk_data.shape, dtype=bool)
        msk_temp[msk_data == i] = True
        ctr_mass = center_of_mass(msk_temp)
        if world:
            ctr_mass = msk.affine[:3, :3].dot(ctr_mass) + msk.affine[:3, 3]
            ctr_mass = ctr_mass.tolist()
        ctd_list.append([i] + [round(x, decimals) for x in ctr_mass])
    return ctd_list


def calc_centroids_by_index(msk, label_index, decimals=1, world=False):
    # Centroids are in voxel coordinates!
    # world=True: centroids are in world coordinates
    msk_data = np.asanyarray(msk.dataobj, dtype=msk.dataobj.dtype)
    axc = nio.aff2axcodes(msk.affine)
    ctd_list = [axc]
    msk_temp = np.zeros(msk_data.shape, dtype=bool)
    msk_temp[msk_data == label_index] = True
    ctr_mass = center_of_mass(msk_temp)
    if world:
        ctr_mass = msk.affine[:3, :3].dot(ctr_mass) + msk.affine[:3, 3]
        ctr_mass = ctr_mass.tolist()
    ctd_list.append([label_index] + [round(x, decimals) for x in ctr_mass])
    return ctd_list


def centroids_to_dict(ctd_list):
    dict_list = []
    for v in ctd_list:
        if any('nan' in str(v_item) for v_item in v):
            continue  # skipping invalid NaN values
        v_dict = {}
        if isinstance(v, tuple):
            v_dict['direction'] = v
        else:
            v_dict['label'] = int(v[0])
            v_dict['X'] = v[1]
            v_dict['Y'] = v[2]
            v_dict['Z'] = v[3]
        dict_list.append(v_dict)
    return dict_list


def save_centroids(ctd_list, out_path):
    if len(ctd_list) < 2:
        print("[#] Centroids empty, not saved:", out_path)
        return
    json_object = centroids_to_dict(ctd_list)
    # Problem with python 3 and int64 serialisation.

    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError
    with open(out_path, 'w') as f:
        json.dump(json_object, f, default=convert)
    print("[*] Centroids saved:", out_path)


def load_centroids(ctd_path):
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    if 'vertebralCentroids' in dict_list:        # needed for Giles Anduin Output
        dict_list = dict_list['vertebralCentroids']
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):  # skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']])
    return ctd_list


def create_figure(dpi, *planes):
    fig_h = round(2 * planes[0].shape[0] / dpi, 2)
    plane_w = [p.shape[1] for p in planes]
    w = sum(plane_w)
    fig_w = round(2 * w / dpi, 2)
    x_pos = [0]
    for x in plane_w[:-1]:
        x_pos.append(x_pos[-1] + x)
    fig, axs = plt.subplots(1, len(planes), figsize=(fig_w, fig_h))
    for a in axs:
        a.axis('off')
        idx = axs.tolist().index(a)
        a.set_position([x_pos[idx]/w, 0, plane_w[idx]/w, 1])
    return fig, axs


def generate_snapshot(ct_file, seg_file, output_path):
    wdw_sbone = Normalize(vmin=-500, vmax=1300, clip=True)
    dpi = 150
    to_ax = ('I', 'A', 'L')

    img = nib.load(ct_file)
    seg = nib.load(seg_file)

    ctd = calc_centroids(seg)
    img = reorient_to(img, to_ax)
    seg = reorient_to(seg, to_ax)
    ctd = reorient_centroids_to(ctd, seg)

    img_data = img.get_fdata()
    h, w, d = np.shape(img_data)
    msk_data = np.asanyarray(seg.dataobj, dtype=seg.dataobj.dtype)

    # Check if liver mask is present
    auxcenter = None
    for center in ctd:
        if center[0] == 5:
            auxcenter = center
    if auxcenter == None:
        # Choose random label from available labels
        n = random.randint(1, len(ctd) - 1)
        auxcenter = ctd[n]
    # Check for boundary condition, happens when the organ is in the limit of image
    slice = int(auxcenter[2])
    if slice >= w:
        slice = slice - 1
    cor_liver = img_data[:, slice, :]
    cor_liver_mask = msk_data[:, slice, :]
    cor_liver_mask = cor_liver_mask.astype(float)
    cor_liver_mask[cor_liver_mask == 0] = np.nan

    # Check if coronal aorta mask is present
    auxcenter = None
    for center in ctd:
        if center[0] == 7:
            auxcenter = center
    if auxcenter == None:
        n = random.randint(1, len(ctd) - 1)
        auxcenter = ctd[n]
    slice = int(auxcenter[2])
    if slice >= w:
        slice = slice - 1
    cor_aorta = img_data[:, slice, :]
    cor_aorta_mask = msk_data[:, slice, :]
    cor_aorta_mask = cor_aorta_mask.astype(float)
    cor_aorta_mask[cor_aorta_mask == 0] = np.nan

    # Check if sagital L4 mask is present
    auxcenter = None
    for center in ctd:
        if center[0] == 19:
            auxcenter = center
    if auxcenter == None:
        n = random.randint(1, len(ctd) - 1)
        auxcenter = ctd[n]
    slice = int(auxcenter[3])
    if slice >= d:
        slice = slice - 1
    sag_l4 = img_data[:, :, slice]
    sag_l4 = np.fliplr(sag_l4)
    sag_l4_mask = msk_data[:, :, slice]
    sag_l4_mask = np.fliplr(sag_l4_mask)
    sag_l4_mask = sag_l4_mask.astype(float)
    sag_l4_mask[sag_l4_mask == 0] = np.nan

    # Check if axial pancreas mask is present plot
    auxcenter = None
    for center in ctd:
        if center[0] == 10:
            auxcenter = center
    if auxcenter == None:
        n = random.randint(1, len(ctd) - 1)
        auxcenter = ctd[n]
    slice = int(auxcenter[1])
    if slice >= h:
        slice = slice - 1
    ax_pancreas = img_data[slice, :, :]
    ax_pancreas = np.fliplr(np.rot90(ax_pancreas, k=2))
    ax_pancreas_mask = msk_data[slice, :, :]
    ax_pancreas_mask = np.fliplr(np.rot90(ax_pancreas_mask, k=2))
    ax_pancreas_mask = ax_pancreas_mask.astype(float)
    ax_pancreas_mask[ax_pancreas_mask == 0] = np.nan

    # Check if axial colon mask is present
    auxcenter = None
    for center in ctd:
        if center[0] == 57:
            auxcenter = center
    if auxcenter == None:
        n = random.randint(1, len(ctd) - 1)
        auxcenter = ctd[n]
    slice = int(auxcenter[1])
    if slice >= h:
        slice = slice - 1
    ax_colon = img_data[slice, :, :]
    ax_colon = np.fliplr(np.rot90(ax_colon, k=2))
    ax_colon_mask = msk_data[slice, :, :]
    ax_colon_mask = np.fliplr(np.rot90(ax_colon_mask, k=2))

    ax_colon_mask = ax_colon_mask.astype(float)
    ax_colon_mask[ax_colon_mask == 0] = np.nan

    # Check if axial lungs mask is present
    auxcenter = None
    for center in ctd:
        if center[0] == 13:
            auxcenter = center
    if auxcenter == None:
        n = random.randint(1, len(ctd) - 1)
        auxcenter = ctd[n]
    slice = int(auxcenter[1])
    if slice >= h:
        slice = slice - 1
    ax_lungs = img_data[slice, :, :]
    ax_lungs = np.fliplr(np.rot90(ax_lungs, k=2))
    ax_lungs_mask = msk_data[slice, :, :]
    ax_lungs_mask = np.fliplr(np.rot90(ax_lungs_mask, k=2))
    ax_lungs_mask = ax_lungs_mask.astype(float)
    ax_lungs_mask[ax_lungs_mask == 0] = np.nan

    fig, axs = create_figure(dpi, cor_liver, cor_aorta,
                             sag_l4, ax_pancreas, ax_colon, ax_lungs)
    fig.patch.set_facecolor('black')

    axs[0].imshow(cor_liver, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[0].imshow(cor_liver_mask, cmap=cm_itk, alpha=0.5, vmin=1, vmax=200)

    axs[1].imshow(cor_aorta, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[1].imshow(cor_aorta_mask, cmap=cm_itk, alpha=0.5, vmin=1, vmax=200)

    axs[2].imshow(sag_l4, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[2].imshow(sag_l4_mask, cmap=cm_itk, alpha=0.5, vmin=1, vmax=200)

    axs[3].imshow(ax_pancreas, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[3].imshow(ax_pancreas_mask, cmap=cm_itk, alpha=0.5, vmin=1, vmax=200)

    axs[4].imshow(ax_colon, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[4].imshow(ax_colon_mask, cmap=cm_itk, alpha=0.5, vmin=1, vmax=200)

    axs[5].imshow(ax_lungs, cmap=plt.cm.gray, norm=wdw_sbone)
    axs[5].imshow(ax_lungs_mask, cmap=cm_itk, alpha=0.5, vmin=1, vmax=200)

    fig.savefig(output_path)
    plt.close()
