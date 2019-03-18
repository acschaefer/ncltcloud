#!/usr/bin/env python

import numpy as np
import open3d as o3
import os
import pyquaternion as pq
import transforms3d as t3

# Path to dataset and session to use.
datadir = '/mnt/data/datasets/nclt'
session = '2012-01-08'

# Conventions.
eulerdef = 'sxyz'
csvdelimiter = ','

def pose2ht(pose):
    # Transform [x, y, z, r, p, y] pose to 4x4 homogeneous transformation
    # matrix.
    r, p, y = pose[3:]
    return t3.affines.compose(
        pose[:3], t3.euler.euler2mat(r, p, y, eulerdef), np.ones(3))

def interpolate_ht(ht, t, tq):
    # Interpolate 4x4 homogeneous transformation matrices.
    amount = np.clip((tq - t[0]) / np.diff(t), 0.0, 1.0)
    pos = ht[0, :3, 3] + amount * np.diff(ht[:, :3, 3], axis=0).squeeze()
    q = [pq.Quaternion(matrix=m) for m in ht]
    qq = pq.Quaternion.slerp(q[0], q[1], amount=amount)
    return t3.affines.compose(pos, qq.rotation_matrix, np.ones(3))

# Load ground truth trajectory with respect to odometry frame.
posefile = os.path.join(
    datadir, 'ground_truth', 'groundtruth_' + session + '.csv')
posedata = np.genfromtxt(posefile, delimiter=csvdelimiter)
posedata = posedata[np.logical_not(np.any(np.isnan(posedata), 1))]
t_odo = posedata[:, 0]
T_o_r = np.stack([pose2ht(pose) for pose in posedata[:, 1:]])

def get_T_o_r(t):
    # Get ground truth robot pose in odometry frame at specified time.
    i = np.clip(np.argmax(t_odo > t), 1, t_odo.size - 1) + np.array([-1, 0])
    return interpolate_ht(T_o_r[i], t_odo[i], t)

# Create transformation from robot frame to lidar frame.
rpy = np.radians([0.807, 0.166, -90.703])
T_r_v = t3.affines.compose([0.002, -0.004, -0.957], 
    t3.euler.euler2mat(rpy[0], rpy[1], rpy[2], axes=eulerdef), np.ones(3))

# Parse velodyne_sync folder to get file names and time stamps of lidar scans.
veloheadertype = np.dtype({
    'magic': ('<u8', 0),
    'count': ('<u4', 8),
    'utime': ('<u8', 12),
    'pad': ('V4', 20)})
veloheadersize = 24
velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodir = os.path.join(
    datadir, 'velodyne_data', session + '_vel', 'velodyne_sync')
velofiles = [os.path.join(velodir, file) \
    for file in os.listdir(velodir) \
    if os.path.splitext(file)[1] == '.bin']
velofiles.sort()
t_velo = np.array([int(os.path.splitext(os.path.basename(velofile))[0]) \
    for velofile in velofiles])

def get_velo(i):
    # Load velodyne scan from file.
    data = np.fromfile(velofiles[i]).view(velodatatype)
    xyz = np.hstack([data[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0
    return xyz

# Accumulate some point clouds and show them.
# globalcloud = o3.PointCloud()
# for i in range(200):
#     localcloud = o3.PointCloud()
#     localcloud.points = o3.Vector3dVector(get_velo(i))
#     localcloud.transform(get_T_o_r(t_velo[i]))
#     globalcloud.points.extend(localcloud.points)
# o3.draw_geometries([globalcloud])

# Accumulate the raw point clouds and show them.
globalcloud = o3.PointCloud()
velorawfile = os.path.join(
    datadir, 'velodyne_data', session + '_vel', 'velodyne_hits.bin')
with open(velorawfile, 'rb') as file:
    for i in range(100000):
        data = np.array(file.read(veloheadersize))
        header = data.view(veloheadertype)
        if header['magic'] != 0xad9cad9cad9cad9c:
            break
        data = np.fromfile(file, count=header['count']).view(velodatatype)
        xyz = np.hstack(
            [data[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
        xyz = xyz * 0.005 - 100.0
        intensities = data['i'].reshape([-1, 1]) / 255.0

        localcloud = o3.PointCloud()
        localcloud.points = o3.Vector3dVector(xyz)
        localcloud.transform(get_T_o_r(header['utime']).dot(T_r_v))
        globalcloud.points.extend(localcloud.points)
        globalcloud.colors.extend(
            o3.Vector3dVector(np.tile(intensities * 0.8, [1, 3])))
o3.draw_geometries([globalcloud])
