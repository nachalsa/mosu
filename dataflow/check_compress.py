import h5py
with h5py.File('batch_03_00_F_poses.h5', 'r') as f:
    print(f['video_0001']['keypoints_scaled'].compression)