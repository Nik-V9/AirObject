from __future__ import print_function
import sys
sys.path.append('.')
import os
from typing import Optional, Union

import cv2
import numpy as np
import pickle
import torch
from torch.utils import data

from cocoapi.PythonAPI.pycocotools.ytvos import YTVOS

__all__ = ["OVIS"]

def find(lst, key, value):
    indices = []
    ids = []
    for i, dic in enumerate(lst):
        if value in dic[key][0]:
            indices.append(i)
            ids.append(lst[i]['id'])
    return indices, ids

class OVIS(data.Dataset):
    r"""A torch Dataset for loading in `the OVIS dataset <http://songbai.site/ovis/>`_. Will fetch sequences of
    rgb images, instance segmentation labels, SuperPoint features (optional).

    Example of sequence creation from frames with `seqlen=4`, `dilation=1`, `stride=3`, and `start=2`:

    .. code-block::


                                            sequence0
                        ┎───────────────┲───────────────┲───────────────┒
                        |               |               |               |
        frame0  frame1  frame2  frame3  frame4  frame5  frame6  frame7  frame8  frame9  frame10  frame11 ...
                                                |               |               |                |
                                                └───────────────┵───────────────┵────────────────┚
                                                                    sequence1


    Args:
        basedir (str): Path to the base directory containing the directories from OVIS.
        instance_path (str): Path to the instances .json file
        videos (str or tuple of str): Videos to use from sequences (used for creating train/val/test splits). Can
            be path to a `.txt` file where each line is a Video Seqeunce name, a tuple of scene names.
        seqlen (int): Number of frames to use for each sequence of frames. Default: 4
        dilation (int or None): Number of (original video's) frames to skip between two consecutive
            frames in the extracted sequence. See above example if unsure.
            If None, will set `dilation = 0`. Default: None
        stride (int or None): Number of frames between the first frames of two consecutive extracted sequences.
            See above example if unsure. If None, will set `stride = seqlen * (dilation + 1)`
            (non-overlapping sequences). Default: None
        start (int or None): Index of the frame from which to start extracting sequences for every video.
            If None, will start from the first frame. Default: None
        end (int): Index of the frame at which to stop extracting sequences for every video.
            If None, will continue extracting frames until the end of the video. Default: None
        height (int): Spatial height to resize frames to. Default: 480
        width (int): Spatial width to resize frames to. Default: 640
        return_seg (bool): Determines whether to return instance segmentation labels. Default: True
        return_points (bool): Determines whether to return SuperPoint Features. Default: False
        return_videonames (bool): Determines whether to return videonames for the sequences. Default: False

    """

    def __init__(
        self,
        basedir: str,
        instance_path: str,
        videos: Union[tuple, str, None],
        seqlen: int = 4,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        height: int = 480,
        width: int = 640,
        *,
        return_img: bool = True,
        return_seg: bool = True,
        return_points: bool = False,
        return_videonames: bool = False,
    ):
        super(OVIS, self).__init__()

        self.basedir = os.path.normpath(basedir)
        self.instance_path = instance_path
        if not os.path.isdir(self.basedir):
            raise ValueError("Base Directory: {} doesn't exist".format(basedir))

        self.height = height
        self.width = width

        self.return_img = return_img
        self.return_seg = return_seg
        self.return_points = return_points
        self.return_videonames = return_videonames

        if not isinstance(seqlen, int):
            raise TypeError("seqlen must be int. Got {0}.".format(type(seqlen)))
        if not (isinstance(stride, int) or stride is None):
            raise TypeError("stride must be int or None. Got {0}.".format(type(stride)))
        if not (isinstance(dilation, int) or dilation is None):
            raise TypeError(
                "dilation must be int or None. Got {0}.".format(type(dilation))
            )
        dilation = dilation if dilation is not None else 0
        stride = stride if stride is not None else seqlen * (dilation + 1)
        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation
        if seqlen < 0:
            raise ValueError("seqlen must be positive. Got {0}.".format(seqlen))
        if dilation < 0:
            raise ValueError('"dilation" must be positive. Got {0}.'.format(dilation))
        if stride < 0:
            raise ValueError("stride must be positive. Got {0}.".format(stride))

        if not (isinstance(start, int) or start is None):
            raise TypeError("start must be int or None. Got {0}.".format(type(start)))
        if not (isinstance(end, int) or end is None):
            raise TypeError("end must be int or None. Got {0}.".format(type(end)))
        start = start if start is not None else 0
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end is None or end > start):
            raise ValueError(
                "end ({0}) must be None or greater than start ({1})".format(end, start)
            )

        # videos should be a tuple
        if isinstance(videos, str):
            if os.path.isfile(videos):
                with open(videos, "r") as f:
                    videos = tuple(f.read().split("\n"))
            else:
                raise ValueError("incorrect filename: {} doesn't exist".format(videos))
        elif not (isinstance(videos, tuple)):
            msg = "videos should either be path to split.txt or tuple of videos, but was of type %r instead"
            raise TypeError(msg % type(videos))

        self.ovis = YTVOS(self.instance_path)

        self.RGB_data = []
        self.Seg_data = []
        self.Points_data = []
        self.Videonames_data = []

        idx = np.arange(self.seqlen) * (self.dilation + 1)

        for video in videos:

            # Identfy video id
            vid_ind, vid_id = find(self.ovis.dataset['videos'], 'file_names', video)

            rgbdir = os.path.join(self.basedir, 'train/')
            rgb_list = [os.path.join(rgbdir, x) for x in self.ovis.dataset['videos'][vid_ind[0]]['file_names']]

            if self.return_points:
                pointsdir = os.path.join(self.basedir, 'points/')
                points_list = [os.path.join(pointsdir, x.replace('.jpg','.pkl')) for x in self.ovis.dataset['videos'][vid_ind[0]]['file_names']]

            video_len = len(rgb_list)

            for start_index in range(self.start, video_len, self.stride):
                if start_index + idx[-1] >= video_len:
                    break
                inds = start_index + idx
                self.RGB_data.append([rgb_list[ind] for ind in inds])

                if self.return_seg:
                    seg_info = {}
                    seg_info['vid_id'] = vid_id
                    seg_info['inds'] = inds
                    self.Seg_data.append(seg_info)

                if self.return_points:
                    self.Points_data.append([points_list[ind] for ind in inds])

                if self.return_videonames:
                    self.Videonames_data.append(video)

        self.num_sequences = len(self.RGB_data)

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

    def __getitem__(self, idx: int):
        r"""Returns the data from the sequence at index idx.

        Returns:
            color_seq (torch.Tensor): Sequence of grayscale rgb images of each frame
            seg_seq (torch.Tensor): Sequence of instance segmentation labels for objects present in the frames
            points_seq (torch.Tensor): Sequence of SuperPoint Features
            videoname (str): Videoname of Sequence

        Shape:
            - color_seq: :math:`(L, 3, H, W)` where `L` denotes sequence length
            - seg_seq: : "math: List of per frame instance segmentations with length `L`
            - points_seq: "math: List of SuperPoint Features with length `L`
        """

        # Read in the color info.
        if self.return_img:
            color_seq_path = self.RGB_data[idx]
        if self.return_points:
            points_seq_path = self.Points_data[idx]

        color_seq, points_seq = [], []

        for i in range(self.seqlen):
          
            if self.return_img:

                image = cv2.imread(color_seq_path[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                image = torch.from_numpy(image).type(torch.float16)
                image = image.permute(2,0,1)
                image /= 255
                color_seq.append(image)

            if self.return_points:
                with open(points_seq_path[i],'rb') as fp:
                    points = pickle.load(fp)
                points_seq.append(points)

        output = []
        if self.return_img:
            color_seq = torch.stack(color_seq, 0).float()
            output.append(color_seq)

        if self.return_seg:
            seg_info = self.Seg_data[idx]
            
            # Annotations info
            anns_ids = self.ovis.getAnnIds(vidIds=seg_info['vid_id'])
            anns = self.ovis.loadAnns(anns_ids)
            
            seg_seq = []
            for ind in seg_info['inds']:
                frame_ann = []
                for a in range(len(anns_ids)):
                    ann = {}
                    ann['obj_id'] = anns_ids[a]
                    try:
                      ann_mask = self.ovis.annToMask(anns[a], ind)
                    except:
                      continue
                    ann['ann_mask'] = ann_mask
                    frame_ann.append(ann)
                seg_seq.append(frame_ann)
            output.append(seg_seq)

        if self.return_points:
            output.append(points_seq)

        if self.return_videonames:
            output.append(self.Videonames_data[idx])

        return tuple(output)