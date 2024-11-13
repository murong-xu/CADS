import numpy as np
from dataset_utils.mappings import replacements


def remove_no_label_slides_gt(gt_array):
    """
    For evaluation on Saros: should use the real GT annotations (discontinuous). Remove the un-annotated label slices. 
    """
    _, _, d = np.shape(gt_array)
    # Finding slices along the third dimension where all values are 255
    slices_with_all_255 = [i for i in range(
        d) if np.all(gt_array[:, :, i] == 255)]
    filtered_gt = np.delete(gt_array, slices_with_all_255, axis=2)
    return filtered_gt, slices_with_all_255


def remove_no_label_slides_pred(pred_array, slices_with_all_255):
    """
    For evaluation on Saros: should use the real GT annotations (discontinuous). Remove the un-annotated label slices. 
    """
    filtered_pred = np.delete(pred_array, slices_with_all_255, axis=2)
    return filtered_pred


class PostprocessingMetric():
    """
    gt: taken from datasets (having GTs)
    pseudo: prediction maps 551-559
    """

    def __init__(self, datasetname: str):
        self.dataset = datasetname
        if self.dataset == "0001_visceral_gc" or self.dataset == "0002_visceral_sc":
            self._postprocess = self._postprocess_visceral
            self.process_gt = self._skip_gt
            self.process_pseudo = self._process_pesudo
        elif self.dataset == "0003_kits21":
            self._postprocess = self._postprocess_kits21
            self.process_gt = self._process_gt
            self.process_pseudo = self._process_pseudo_kits21
        elif self.dataset == "0004_lits":
            self._postprocess = self._postprocess_lits
            self.process_gt = self._process_gt
            self.process_pseudo = self._skip_pseudo
        elif self.dataset == "0008_ctorg":
            self._postprocess = self._postprocess_ctorg
            self.process_gt = self._skip_gt
            self.process_pseudo = self._process_pesudo
        elif self.dataset == "0009_abdomenct1k":
            self._postprocess = self._postprocess_abdomenct1k
            self.process_gt = self._skip_gt
            self.process_pseudo = self._process_pesudo
        elif self.dataset == "0034_empire":
            self._postprocess = self._postprocess_empire
            self.process_gt = self._skip_gt
            self.process_pseudo = self._process_pesudo
        elif self.dataset == "0040_saros":
            self.process_gt = self._process_gt_saros
            self.process_pseudo = self._process_pseudo_saros
        else:
            print("No postprocessing for this dataset needs to be done.")
            self.process_pseudo = self._skip_pseudo
            self.process_gt = self._skip_gt

    def _skip_pseudo(self, part, labelmap, label):
        return labelmap, label

    def _skip_gt(self, labelmap, label):
        return labelmap, label

    def _process_pesudo(self, part, labelmap, label):
        labelmap_copy = dict(labelmap)
        if part == 551:
            labelmap_copy, label = self._postprocess(labelmap_copy, label)
        return labelmap_copy, label

    def _process_gt(self, labelmap, label):
        labelmap_copy = dict(labelmap)
        labelmap_copy, label = self._postprocess(labelmap_copy, label)
        return labelmap_copy, label

    def _postprocess_visceral(self, pseudo_labelmap, pseudo):
        """
        pseudo: merge 15+16+17 (right lobes)  -> right lung, merge 13+14 (left lobes)-> left lung
        """
        del pseudo_labelmap[13]
        del pseudo_labelmap[14]
        del pseudo_labelmap[15]
        del pseudo_labelmap[16]
        del pseudo_labelmap[17]
        pseudo_labelmap[13] = "left lung"
        pseudo_labelmap[14] = "right lung"
        pseudo[pseudo == 13] = 13
        pseudo[pseudo == 14] = 13
        pseudo[pseudo == 15] = 14
        pseudo[pseudo == 16] = 14
        pseudo[pseudo == 17] = 14
        return pseudo_labelmap, pseudo

    def _postprocess_kits21(self, gt_labelmap, gt):
        """
        gt: merge lesion(2) & cyst(3) to kidney(1)
        """
        del gt_labelmap[2]
        del gt_labelmap[3]
        gt[gt == 2] = 1
        gt[gt == 3] = 1
        return gt_labelmap, gt

    def _process_pseudo_kits21(self, part, pseudo_labelmap, pseudo):
        """
        pesudo: merge 2+3->kidney
        """
        pseudo_labelmap_copy = dict(pseudo_labelmap)
        if part == 551:
            del pseudo_labelmap_copy[2]
            del pseudo_labelmap_copy[3]
            pseudo_labelmap_copy[2] = "kidney"
            pseudo[pseudo == 2] = 2
            pseudo[pseudo == 3] = 2
        return pseudo_labelmap_copy, pseudo

    def _process_pseudo_saros(self, part, pseudo_labelmap, pseudo):
        """
        Remove slices with no annotations.
        """
        if part in [553, 559]:
            pseudo = remove_no_label_slides_pred(
                pseudo, self.slices_with_all_255)
        return pseudo_labelmap, pseudo

    def _process_gt_saros(self, gt_labelmap, gt):
        """
        Remove "nolabel" in Saros GT labelmap.
        Remove slices with no annotations.
        """
        gt_labelmap_copy = dict(gt_labelmap)
        # del gt_labelmap_copy[255]  # after removing axal slices with 255s, the remaining ones still can contain 255 labels

        gt, self.slices_with_all_255 = remove_no_label_slides_gt(gt)

        return gt_labelmap_copy, gt

    def _postprocess_lits(self, gt_labelmap, gt):
        """
        gt: merge lesion(2) to liver(1)
        """
        del gt_labelmap[2]
        gt[gt == 2] = 1
        return gt_labelmap, gt

    def _postprocess_ctorg(self, pseudo_labelmap, pseudo):
        """
        pesudo: merge 13+14+15+16+17 (lung lobes)->lungs, merge 2+3->kidneys
        """
        del pseudo_labelmap[13]
        del pseudo_labelmap[14]
        del pseudo_labelmap[15]
        del pseudo_labelmap[16]
        del pseudo_labelmap[17]
        pseudo_labelmap[13] = "lungs"
        pseudo[pseudo == 13] = 13
        pseudo[pseudo == 14] = 13
        pseudo[pseudo == 15] = 13
        pseudo[pseudo == 16] = 13
        pseudo[pseudo == 17] = 13
        del pseudo_labelmap[2]
        del pseudo_labelmap[3]
        pseudo_labelmap[2] = "kidneys"
        pseudo[pseudo == 2] = 2
        pseudo[pseudo == 3] = 2
        return pseudo_labelmap, pseudo

    def _postprocess_abdomenct1k(self, pseudo_labelmap, pseudo):
        """
        pseudo: merge 2+3->kidney
        """
        del pseudo_labelmap[2]
        del pseudo_labelmap[3]
        pseudo_labelmap[2] = "kidney"
        pseudo[pseudo == 2] = 2
        pseudo[pseudo == 3] = 2
        return pseudo_labelmap, pseudo

    def _postprocess_empire(self, pseudo_labelmap, pseudo):
        """
        pseudo: merge 13+14+15+16+17 (lung lobes)->lungs
        """
        del pseudo_labelmap[13]
        del pseudo_labelmap[14]
        del pseudo_labelmap[15]
        del pseudo_labelmap[16]
        del pseudo_labelmap[17]
        pseudo_labelmap[13] = "lungs"
        pseudo[pseudo == 13] = 13
        pseudo[pseudo == 14] = 13
        pseudo[pseudo == 15] = 13
        pseudo[pseudo == 16] = 13
        pseudo[pseudo == 17] = 13
        return pseudo_labelmap, pseudo


class RecalculateAvgOrganVolume():
    """
    gt: taken from datasets (having GTs)
    Goal: to be able to used in combination with our models 551-559's available targets
    """

    def __init__(self, avg_volume: dict, tolerance: float):
        self.replacements = replacements
        self.avg_volume = avg_volume
        self.avg_volume_new = dict(self.avg_volume)
        self.tolerance = tolerance
        self.recalculate = self._recalculate

    def _replace_labelname(self):
        for part in self.avg_volume_new.keys():
            volume_dict = self.avg_volume_new[part]
            for k, _ in list(volume_dict.items()):
                self.avg_volume_new[part][self.replacements.get(
                    k, k)] = self.avg_volume_new[part].pop(k)

    def _set_tolerance(self):
        for part in self.avg_volume_new.keys():
            self.avg_volume_new[part].update((key, round(
                value * self.tolerance)) for key, value in self.avg_volume_new[part].items())

    def _recalculate(self):
        self._replace_labelname()
        self._set_tolerance()
        return self.avg_volume_new
