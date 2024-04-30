import os
import pickle
import time

import numpy as np
from torch.utils.data import Dataset

from data.scannet.model_util_scannet import rotate_aligned_boxes_along_axis
from lib.config import CONF
from lib.dataset import ScannetQADataset, MAX_NUM_OBJ, DC, MULTIVIEW_DATA, MEAN_COLOR_RGB, get_answer_score
from utils.pc_utils import rotx, roty, rotz, random_sampling


class MixedDataset(Dataset):
    def __init__(self, scannet_dataset: ScannetQADataset):
        self.scannet_dataset = scannet_dataset

    def __getitem__(self, idx):
        start = time.time()
        scene_id = self.scannet_dataset.scanqa[idx]['scene_id']
        if self.scannet_dataset.split != 'test':
            object_ids = self.scannet_dataset.scanqa[idx]['object_ids']
            object_names = [' '.join(object_name.split('_')) for object_name in self.scannet_dataset.scanqa[idx]['object_names']]
        else:
            object_ids = None
            object_names = None

        question_id = self.scannet_dataset.scanqa[idx]['question_id']
        answers = self.scannet_dataset.scanqa[idx].get('answers', [])

        answer_cats = np.zeros(self.scannet_dataset.num_answers)
        answer_inds = [self.scannet_dataset.answer_vocab.stoi(answer) for answer in answers]

        if self.scannet_dataset.answer_counter is not None:
            answer_cat_scores = np.zeros(self.scannet_dataset.num_answers)
            for answer, answer_ind in zip(answers, answer_inds):
                if answer_ind < 0:
                    continue
                answer_cats[answer_ind] = 1
                answer_cat_score = get_answer_score(self.scannet_dataset.answer_counter.get(answer, 0))
                answer_cat_scores[answer_ind] = answer_cat_score

            if not self.scannet_dataset.use_unanswerable:
                assert answer_cats.sum() > 0
                assert answer_cat_scores.sum() > 0
        else:
            raise NotImplementedError

        answer_cat = answer_cats.argmax()

        #
        # get language features
        #
        if self.scannet_dataset.use_bert_embeds:
            lang_feat = self.scannet_dataset.lang[scene_id][question_id]
            lang_feat['input_ids'] = lang_feat['input_ids'].astype(np.int64)
            lang_feat['attention_mask'] = lang_feat['attention_mask'].astype(np.float32)
            if 'token_type_ids' in lang_feat:
                lang_feat['token_type_ids'] = lang_feat['token_type_ids'].astype(np.int64)
            lang_len = self.scannet_dataset.scanqa[idx]['token']['input_ids'].shape[1]
        else:
            lang_feat = self.scannet_dataset.lang[scene_id][question_id]
            lang_len = len(self.scannet_dataset.scanqa[idx]['token'])

        lang_len = lang_len if lang_len <= CONF.TRAIN.MAX_TEXT_LEN else CONF.TRAIN.MAX_TEXT_LEN
        #
        # get point cloud features
        #
        mesh_vertices = self.scannet_dataset.scene_data[scene_id]['mesh_vertices']
        instance_labels = self.scannet_dataset.scene_data[scene_id]['instance_labels']
        semantic_labels = self.scannet_dataset.scene_data[scene_id]['semantic_labels']
        instance_bboxes = self.scannet_dataset.scene_data[scene_id]['instance_bboxes']

        if not self.scannet_dataset.use_color:
            point_cloud = mesh_vertices[:, 0:3]
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = (point_cloud[:, 3:6] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:6]

        if self.scannet_dataset.use_normal:
            normals = mesh_vertices[:, 6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)  # p (50000, 7)

        if self.scannet_dataset.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        '''
        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA + '.hdf5', 'r', libver='latest')
            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)
        '''

        # '''
        if self.scannet_dataset.use_multiview:
            # load multiview database
            enet_feats_file = os.path.join(MULTIVIEW_DATA, scene_id) + '.pkl'
            multiview = pickle.load(open(enet_feats_file, 'rb'))
            point_cloud = np.concatenate([point_cloud, multiview], 1)  # p (50000, 135)
        # '''

        point_cloud, choices = random_sampling(point_cloud, self.scannet_dataset.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        ref_box_label = np.zeros(MAX_NUM_OBJ)  # bbox label for reference target

        ref_center_label = np.zeros(3)  # bbox center for reference target
        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3)  # bbox size residual for reference target

        if self.scannet_dataset.split != 'test':
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]

            point_votes = np.zeros([self.scannet_dataset.num_points, 3])
            point_votes_mask = np.zeros(self.scannet_dataset.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------
            if self.scannet_dataset.augment and not self.scannet_dataset.debug:
                if np.random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if np.random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                    # Rotation along X-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'x')

                # Rotation along Y-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'y')

                # Rotation along up-axis/Z-axis
                rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, 'z')

                # Translation
                point_cloud, target_bboxes = self.scannet_dataset._translate(point_cloud, target_bboxes)

            # compute votes *AFTER* augmentation
            # generate votes
            # Note: since there's no map between bbox instance labels and
            # pc instance_labels (it had been filtered
            # in the data preparation step) we'll compute the instance bbox
            # from the points sharing the same instance label.
            for i_instance in np.unique(instance_labels):
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind, :3]
                    center = 0.5 * (x.min(0) + x.max(0))
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3))  # make 3 votes identical

            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox, -2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind, :]

            # construct the reference target label for each bbox
            ref_box_label = np.zeros(MAX_NUM_OBJ)

            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                if gt_id == object_ids[0]:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]

            assert ref_box_label.sum() > 0
        else:
            num_bbox = 1
            point_votes = np.zeros([self.scannet_dataset.num_points, 9])  # make 3 votes identical
            point_votes_mask = np.zeros(self.scannet_dataset.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in
                                                instance_bboxes[:, -2][0:num_bbox]]
        except KeyError:
            pass

        object_name = None if object_names is None else object_names[0]
        object_cat = self.scannet_dataset.raw2label[object_name] if object_name in self.scannet_dataset.raw2label else 17

        data_dict = {}
        if self.scannet_dataset.use_bert_embeds:
            data_dict['lang_feat'] = lang_feat
        else:
            data_dict['lang_feat'] = lang_feat.astype(np.float32)  # language feature vectors
        data_dict['point_clouds'] = point_cloud.astype(np.float32)  # point cloud data including features
        data_dict['lang_len'] = np.array(lang_len).astype(np.int64)  # length of each description
        data_dict['center_label'] = target_bboxes.astype(np.float32)[:,
                                    0:3]  # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict['heading_class_label'] = angle_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict['heading_residual_label'] = angle_residuals.astype(np.float32)  # (MAX_NUM_OBJ,)
        data_dict['size_class_label'] = size_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict['size_residual_label'] = size_residuals.astype(np.float32)  # (MAX_NUM_OBJ, 3)
        data_dict['num_bbox'] = np.array(num_bbox).astype(np.int64)
        data_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)  # (MAX_NUM_OBJ,) semantic class index
        data_dict['box_label_mask'] = target_bboxes_mask.astype(
            np.float32)  # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict['vote_label'] = point_votes.astype(np.float32)  #
        data_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)  # point_obj_mask (gf3d)
        data_dict['scan_idx'] = np.array(idx).astype(np.int64)
        data_dict['pcl_color'] = pcl_color
        data_dict['ref_box_label'] = ref_box_label.astype(
            np.int64)  # (MAX_NUM_OBJ,) # 0/1 reference labels for each object bbox

        data_dict['ref_center_label'] = ref_center_label.astype(np.float32)  # (3,)
        data_dict['ref_heading_class_label'] = np.array(int(ref_heading_class_label)).astype(
            np.int64)  # (MAX_NUM_OBJ,)
        data_dict['ref_heading_residual_label'] = np.array(int(ref_heading_residual_label)).astype(
            np.int64)  # (MAX_NUM_OBJ,)
        data_dict['ref_size_class_label'] = np.array(int(ref_size_class_label)).astype(np.int64)  # (MAX_NUM_OBJ,)
        data_dict['ref_size_residual_label'] = ref_size_residual_label.astype(np.float32)
        data_dict['object_cat'] = np.array(object_cat).astype(np.int64)

        data_dict['scene_id'] = np.array(int(self.scannet_dataset.scene_id_to_number[scene_id])).astype(np.int64)
        if type(question_id) == str:
            data_dict['question_id'] = np.array(int(question_id.split('-')[-1])).astype(np.int64)
        else:
            data_dict['question_id'] = np.array(int(question_id)).astype(np.int64)
        data_dict['pcl_color'] = pcl_color
        data_dict['load_time'] = time.time() - start
        data_dict['answer_cat'] = np.array(int(answer_cat)).astype(np.int64)  # 1
        data_dict['answer_cats'] = answer_cats.astype(np.int64)  # num_answers
        if self.scannet_dataset.answer_cls_loss == 'bce' and self.scannet_dataset.answer_counter is not None:
            data_dict['answer_cat_scores'] = answer_cat_scores.astype(np.float32)  # num_answers
        return data_dict
