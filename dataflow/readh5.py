import h5py

# 파일 열기
with h5py.File('batch_03_01_F_poses.h5', 'r') as f:
    # 최상위 그룹/데이터셋 목록 확인
    print("Keys:", list(f.keys()))
    
    # 재귀적으로 모든 구조 탐색
    def print_structure(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Type: {obj.dtype}")
    
    f.visititems(print_structure)