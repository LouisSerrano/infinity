# -*- encoding: utf-8 -*-
"""
@File    :   parse_tfrecord.py
@Author  :   jianglx
@Version :   1.0
@Contact :   jianglx@whu.edu.cn
"""
import pdb

# 解析tfrecord解析数据，存为hdf5文件
import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py
from data.deepmind.fpc import FPC
from torch_geometric.loader import DataLoader


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


if __name__ == "__main__":
    # dataset_dir = "/home/leon/PycharmProjects/INR_domain_decomposition/data/deepmind"
    # dataset_dir = "/data/migus/INR_domain_decomposition/"
    # batch_size = 20
    # dataset_fpc = FPC(dataset_dir=dataset_dir, split='train', max_epochs=50)
    # train_loader = DataLoader(dataset=dataset_fpc, batch_size=batch_size, num_workers=10)
    # pdb.set_trace()

    # tf.enable_resource_variables()
    # tf.enable_eager_execution()

    data_file = "airfoil"
    tf_datasetPath = "/data/migus/inr_domain_decomposition/" + data_file
    os.makedirs("/data/migus/inr_domain_decomposition/", exist_ok=True)

    for split in ["train", "test", "valid"]:
        ds = load_dataset(tf_datasetPath, split)
        save_path = (
            "/data/migus/inr_domain_decomposition/" + split + "_" + data_file + ".h5"
        )
        f = h5py.File(save_path, "w")
        print(save_path)

        for index, d in enumerate(ds):
            pos = d["mesh_pos"].numpy()
            node_type = d["node_type"].numpy()
            cells = d["cells"].numpy()

            if data_file == "deforming_plate":
                world_pos = d["world_pos"].numpy()
                stress = d["stress"].numpy()
                data = ("pos", "node_type", "world_pos", "cells", "stress")
            elif data_file == "airfoil":
                density = d["density"].numpy()
                pressure = d["pressure"].numpy()
                velocity = d["velocity"].numpy()
                data = ("pos", "node_type", "cells", "density", "velocity", "pressure")
            elif data_file == "cylinder_flow":
                pressure = d["pressure"].numpy()
                velocity = d["velocity"].numpy()
                data = ("pos", "node_type", "cells", "pressure", "velocity")

            # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
            g = f.create_group(str(index))
            for k in data:
                g[k] = eval(k)

            print(index)
        f.close()
