import os
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join, exists
from os import listdir, makedirs


class Feeder(Dataset):
    """
    Dataset loader for distilled R2ET.
    Provides:
      quat_src, skel_src, shape_src,
      delta_qs, delta_qg, mask.
    """

    def __init__(self, data_path, stats_path, max_length=None, num_joint=22, mode='train'):
        super().__init__()
        self.data_path = data_path
        self.stats_path = stats_path
        self.max_length = max_length
        self.num_joint = num_joint
        self.mode = mode

        self.parents = np.array(
            [-1, 0, 1, 2, 3, 4,
              0, 6, 7, 8,
              0, 10, 11, 12,
              3, 14, 15, 16,
              3, 18, 19, 20],
            dtype=np.int64
        )

        self.load_data()

    def load_data(self):
        all_quat = []
        all_skel = []
        all_shape = []
        all_delta_qs = []
        all_delta_qg = []
        all_quatB_rt = []

        #if self.mode == "train":
        #    source_list = ["Aj", "Big", "Goblin", "Kaya", "Peasant", "Warrok"]
        #else:
        #    source_list = ["Amy"]

        if self.mode == "train":
            source_list = ["abe", "adam", "alien soldier", "crypto", "demon t wiezzorek", "exo gray", "james", "leonard"]
        else:
            source_list = ["steve"]

        target = "Claire"
        self.shape_index = []

        for i, source in enumerate(source_list):
            source_path = join(self.data_path, "source", source)
            target_path = join(self.data_path, "target", target, source)
            shape_path = join(self.data_path, "shape", source + ".npz")

            shape_npz = np.load(shape_path)
            full_width = shape_npz['full_width'].astype(np.single)
            joint_shape = shape_npz['joint_shape'].astype(np.single)

            shape_vector = np.divide(joint_shape, full_width[None, :])
            all_shape.append(shape_vector)

            motion_list = sorted(
                [
                    os.path.splitext(f)[0]
                    for f in listdir(target_path)
                    if not f.startswith(".") and f.endswith(".npz")
                ]
            )

            for motion in motion_list:
                quat = np.load(join(source_path, motion + "_quat.npy"))
                all_quat.append(quat)

                skel = np.load(join(source_path, motion + "_skel.npy"))[0]
                all_skel.append(skel)

                tgt = np.load(join(target_path, motion + ".npz"))
                all_delta_qs.append(tgt['delta_qs'][0])
                all_delta_qg.append(tgt['delta_qg'][0])
                all_quatB_rt.append(tgt['quatB_rt'][0])

                self.shape_index.append(i)

        train_quat = all_quat
        train_delta_qs = all_delta_qs
        train_delta_qg = all_delta_qg
        train_quatB_rt = all_quatB_rt

        if not exists(join(self.stats_path, 'stats.npz')):
            # all_cat = np.concatenate(all_quat + all_delta_qs + all_delta_qg + all_quatB_rt, axis=0)
            all_cat = np.concatenate(all_quat, axis=0)
            self.q_mean = all_cat.mean(axis=0)
            self.q_std = all_cat.std(axis=0)
            self.q_std[self.q_std < 1e-8] = 1e-8

            train_skel_np = np.stack(all_skel, axis=0)
            self.skel_mean = train_skel_np.mean(axis=0)
            self.skel_std = train_skel_np.std(axis=0)
            self.skel_std[self.skel_std < 1e-8] = 1e-8

            train_shape_np = np.stack(all_shape, axis=0)
            self.shape_mean = train_shape_np.mean(axis=0)
            self.shape_std = train_shape_np.std(axis=0)
            self.shape_std[self.shape_std < 1e-8] = 1e-8

            makedirs(self.stats_path, exist_ok=True)
            np.savez(
                join(self.stats_path, 'stats.npz'),
                q_mean=self.q_mean, q_std=self.q_std,
                skel_mean=self.skel_mean, skel_std=self.skel_std,
                shape_mean=self.shape_mean, shape_std=self.shape_std
            )
        else:
            stats = np.load(join(self.stats_path, 'stats.npz'))
            self.q_mean = stats['q_mean']
            self.q_std = stats['q_std']
            self.skel_mean = stats['skel_mean']
            self.skel_std = stats['skel_std']
            self.shape_mean = stats['shape_mean']
            self.shape_std = stats['shape_std']

        for i in range(len(train_quat)):
            train_quat[i] = (train_quat[i] - self.q_mean) / self.q_std
            #train_delta_qs[i] = (train_delta_qs[i] - self.q_mean) / self.q_std
            #train_delta_qg[i] = (train_delta_qg[i] - self.q_mean) / self.q_std
            #train_quatB_rt[i] = (train_quatB_rt[i] - self.q_mean) / self.q_std
            all_skel[i] = (all_skel[i] - self.skel_mean) / self.skel_std

        train_shape_np = (np.stack(all_shape, axis=0) - self.shape_mean) / self.shape_std
        self.train_quat = train_quat
        self.train_delta_qs = train_delta_qs
        self.train_delta_qg = train_delta_qg
        self.train_quatB_rt = train_quatB_rt
        self.train_skel = all_skel
        self.train_shape = train_shape_np

    def __len__(self):
        return len(self.train_skel)

    def __getitem__(self, idx):
        quat = self.train_quat[idx]
        skel = self.train_skel[idx]
        shape = self.train_shape[self.shape_index[idx]]
        dqs = self.train_delta_qs[idx]
        dqg = self.train_delta_qg[idx]
        qrt = self.train_quatB_rt[idx]

        T, J, _ = quat.shape
        max_len = self.max_length if self.max_length else T

        if self.max_length is None:
            stidx = 0
        else:
            high = max(0, T - max_len)
            stidx = np.random.randint(0, high + 1) if high > 0 else 0

        quat_slice = quat[stidx:stidx + max_len]
        dqs_slice = dqs[stidx:stidx + max_len]
        dqg_slice = dqg[stidx:stidx + max_len]
        qrt_slice = qrt[stidx:stidx + max_len]

        valid_len = quat_slice.shape[0]
        mask = np.zeros((max_len,), dtype=np.float32)
        mask[:valid_len] = 1.0

        if valid_len < max_len:
            pad_T = max_len - valid_len
            id_quat = np.zeros((pad_T, J, 4), dtype=np.float32)
            id_quat[:, :, 0] = 1.0
            id_quat = (id_quat - self.q_mean) / self.q_std

            quat_slice = np.concatenate([quat_slice, id_quat], axis=0)
            dqs_slice = np.concatenate([dqs_slice, id_quat], axis=0)
            dqg_slice = np.concatenate([dqg_slice, id_quat], axis=0)
            qrt_slice = np.concatenate([qrt_slice, id_quat], axis=0)

        quat_t = torch.from_numpy(quat_slice)
        skel_t = torch.from_numpy(skel)
        shape_t = torch.from_numpy(shape)
        dqs_t = torch.from_numpy(dqs_slice)
        dqg_t = torch.from_numpy(dqg_slice)
        qrt_t = torch.from_numpy(qrt_slice)
        mask_t = torch.from_numpy(mask)

        return quat_t, skel_t, shape_t, dqs_t, dqg_t, qrt_t, mask_t, self.q_mean, self.q_std
