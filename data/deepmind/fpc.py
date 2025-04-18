import pdb

from torch.utils.data import IterableDataset
import os, numpy as np
import os.path as osp
import h5py
from torch_geometric.data import Data
import torch
import math
import time


class FPCBase:
    def __init__(self, dataset, max_epochs=1, files=None, scale_factor=1):
        self.open_tra_num = 10
        self.file_handle = files
        self.shuffle_file()

        self.dataset = dataset  # between deforming_plate, airfoil and cylinder_flow
        if self.dataset == "deforming_plate":
            self.data_keys = ("pos", "node_type", "cells", "world_pos", "stress")
            self.target_key = "stress"
        elif self.dataset == "airfoil":
            self.data_keys = (
                "pos",
                "node_type",
                "cells",
                "pressure",
                "velocity",
                "density",
            )
            self.target_key = "velocity"
        elif self.dataset == "cylinder_flow":
            self.data_keys = ("pos", "node_type", "cells", "pressure", "velocity")
            self.target_key = "velocity"
        self.out_keys = list(self.data_keys) + ["time"]

        self.scale_factor = scale_factor

        self.tra_index = 0
        self.epcho_num = 1
        self.tra_readed_index = -1

        # dataset attr
        self.tra_len = 400
        self.time_iterval = 0.01

        self.opened_tra = []
        self.opened_tra_readed_index = {}
        self.opened_tra_readed_random_index = {}
        self.tra_data = {}
        self.max_epochs = max_epochs

    def open_tra(self):
        while len(self.opened_tra) < self.open_tra_num:
            tra_index = self.datasets[self.tra_index]

            if tra_index not in self.opened_tra:
                self.opened_tra.append(tra_index)
                self.opened_tra_readed_index[tra_index] = -1
                self.opened_tra_readed_random_index[tra_index] = np.random.permutation(
                    self.tra_len - 2
                )

            self.tra_index += 1

            if self.check_if_epcho_end():
                self.epcho_end()
                print("Epoch Finished")

    def check_and_close_tra(self):
        to_del = []
        for tra in self.opened_tra:
            if self.opened_tra_readed_index[tra] >= (self.tra_len - 3):
                to_del.append(tra)
        for tra in to_del:
            self.opened_tra.remove(tra)
            try:
                del self.opened_tra_readed_index[tra]
                del self.opened_tra_readed_random_index[tra]
                del self.tra_data[tra]
            except Exception as e:
                print(e)

    def shuffle_file(self):
        datasets = list(self.file_handle.keys())
        np.random.shuffle(datasets)
        self.datasets = datasets

    def epcho_end(self):
        self.tra_index = 0
        self.shuffle_file()
        self.epcho_num = self.epcho_num + 1

    def check_if_epcho_end(self):
        if self.tra_index >= len(self.file_handle):
            return True
        return False

    @staticmethod
    def datas_to_graph(datas, dataset, edge_index=None, scale_factor=1):
        # if edge_index is None: # rollout时第二步不为None，用于避免重复计算
        #     edge_index =  triangles_to_edges_numpy(datas[3])
        #     edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        time_vector = np.ones((datas[0].shape[0], 1)) * datas[-1]
        input_frame = datas[4][0]
        target = datas[4][1]  # velocity or stress depending on data

        # Scaling input frame and target
        input_frame = input_frame / scale_factor
        target = target / scale_factor

        if dataset == "airfoil":
            dens = datas[5]
            press = datas[3]
            node_attr = np.hstack((datas[1], time_vector, dens, press, input_frame))
        elif dataset == "cylinder_flow":
            press = datas[3]
            node_attr = np.hstack((datas[1], time_vector, press, input_frame))
        else:
            node_attr = np.hstack((datas[1], time_vector, input_frame))
        # "node_type, cur_v, pressure, time"
        # new datas: ('pos', 'node_type', 'world_pos', 'cells', 'stress', 'time')
        "node_type, stress, time"
        crds = torch.as_tensor(datas[0], dtype=torch.float)
        # senders = edge_index[0].numpy()
        # receivers = edge_index[1].numpy()
        # crds_diff = crds[senders] - crds[receivers]
        # crds_norm = np.linalg.norm(crds_diff, axis=1, keepdims=True)
        # edge_attr = np.concatenate((crds_diff, crds_norm), axis=1)

        # node_type, cur_v, pressure, time
        node_attr = torch.as_tensor(node_attr, dtype=torch.float32)
        # edge_attr = torch.from_numpy(edge_attr)
        target = torch.from_numpy(target)
        if dataset == "cylinder_flow":
            face = torch.as_tensor([0], dtype=torch.long)
        else:
            face = torch.as_tensor(datas[2], dtype=torch.long)

        g = Data(x=node_attr, face=face, y=target, pos=crds)
        # g = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, y=target, pos=crds)
        return g

    def __next__(self):
        self.check_and_close_tra()
        self.open_tra()

        if self.epcho_num > self.max_epochs:
            raise StopIteration

        selected_tra = np.random.choice(self.opened_tra)

        data = self.tra_data.get(selected_tra, None)
        if data is None:
            data = self.file_handle[selected_tra]
            self.tra_data[selected_tra] = data

        selected_tra_readed_index = self.opened_tra_readed_index[selected_tra]
        selected_frame = self.opened_tra_readed_random_index[selected_tra][
            selected_tra_readed_index + 1
        ]
        self.opened_tra_readed_index[selected_tra] += 1

        datas = []
        for k in self.data_keys:
            if k in [self.target_key]:
                r = np.array(
                    (data[k][selected_frame], data[k][selected_frame + 1]),
                    dtype=np.float32,
                )
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))
        # ("pos", "node_type", "velocity", "cells", "pressure", "time")
        # new : ('pos', 'node_type', 'world_pos', 'cells', 'stress', 'time')
        g = self.datas_to_graph(
            datas, dataset=self.dataset, scale_factor=self.scale_factor
        )

        return g

    def __iter__(self):
        return self


class FPC(IterableDataset):
    def __init__(
        self, max_epochs, dataset_dir, dataset, split="train", scale_factor=1
    ) -> None:
        super().__init__()

        dataset_dir = osp.join(dataset_dir, split + ".h5")
        self.max_epochs = max_epochs
        self.dataset_dir = dataset_dir
        self.scale_factor = scale_factor

        self.dataset = dataset  # between deforming_plate, airfoil and cylinder_flow
        if self.dataset == "deforming_plate":
            self.data_keys = ("pos", "node_type", "cells", "world_pos", "stress")
        elif self.dataset == "airfoil":
            self.data_keys = (
                "pos",
                "node_type",
                "cells",
                "density",
                "velocity",
                "pressure",
            )
        elif self.dataset == "cylinder_flow":
            self.data_keys = ("pos", "node_type", "cells", "pressure", "velocity")

        assert os.path.isfile(dataset_dir), "%s not exist" % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        print("Dataset " + self.dataset_dir + " Initilized")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_handle)
        else:
            per_worker = int(
                math.ceil(len(self.file_handle) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_handle))

        keys = list(self.file_handle.keys())
        keys = keys[iter_start:iter_end]
        files = {k: self.file_handle[k] for k in keys}
        return FPCBase(
            max_epochs=self.max_epochs,
            dataset=self.dataset,
            files=files,
            scale_factor=self.scale_factor,
        )


class FPC_ROLLOUT(IterableDataset):
    def __init__(self, dataset_dir, dataset, split="test", scale_factor=10000):
        dataset_dir = osp.join(dataset_dir, split + ".h5")
        self.scale_factor = scale_factor
        self.dataset = dataset  # between deforming_plate, airfoil and cylinder_flow
        self.dataset_dir = dataset_dir

        assert os.path.isfile(dataset_dir), "%s not exist" % dataset_dir
        self.file_handle = h5py.File(dataset_dir, "r")
        if self.dataset == "deforming_plate":
            self.data_keys = ("pos", "node_type", "cells", "world_pos", "stress")
            self.target_key = "stress"
        elif self.dataset == "airfoil":
            self.data_keys = (
                "pos",
                "node_type",
                "cells",
                "density",
                "velocity",
                "pressure",
            )
            self.target_key = "velocity"
        elif self.dataset == "cylinder_flow":
            self.data_keys = ("pos", "node_type", "cells", "pressure", "velocity")
            self.target_key = "velocity"
        # self.data_keys = ("pos", "node_type", "velocity", "cells", "pressure")
        self.time_iterval = 0.01
        self.load_dataset()

    def load_dataset(self):
        datasets = list(self.file_handle.keys())
        self.datasets = datasets

    def change_file(self, file_index):
        file_index = self.datasets[file_index]
        self.cur_tra = self.file_handle[file_index]
        if self.dataset == "deforming_plate":
            self.cur_targecity_length = self.cur_tra["stress"].shape[0]
        else:
            self.cur_targecity_length = self.cur_tra["velocity"].shape[0]
        self.cur_tragecity_index = 0
        self.edge_index = None

    def __next__(self):
        if self.cur_tragecity_index == (self.cur_targecity_length - 1):
            raise StopIteration

        datas = []
        data = self.cur_tra
        selected_frame = self.cur_tragecity_index

        datas = []
        for k in self.data_keys:
            if k in [self.target_key]:  # ["velocity", "pressure"]:
                r = np.array(
                    (data[k][selected_frame], data[k][selected_frame + 1]),
                    dtype=np.float32,
                )
            else:
                r = data[k][selected_frame]
                if k in ["node_type", "cells"]:
                    r = r.astype(np.int32)
            datas.append(r)
        datas.append(np.array([self.time_iterval * selected_frame], dtype=np.float32))

        self.cur_tragecity_index += 1
        g = FPCBase.datas_to_graph(
            datas, dataset=self.dataset, scale_factor=self.scale_factor
        )
        # self.edge_index = g.edge_index
        return g

    def __iter__(self):
        return self
