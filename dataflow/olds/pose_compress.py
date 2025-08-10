import h5py
import numpy as np
import os
import glob
from tqdm import tqdm

def optimize_h5_file(input_file, output_file):
    """
    H5 파일을 최적화하여 새로운 파일로 저장
    - keypoints_original 삭제
    - keypoints_scaled를 int16으로 변환
    - scores는 그대로 유지 (float32)
    - lzf 압축 적용
    """
    print(f"Processing: {input_file} -> {output_file}")
    
    with h5py.File(input_file, 'r') as src, h5py.File(output_file, 'w') as dst:
        # 각 비디오 그룹 처리
        for video_key in tqdm(src.keys(), desc="Processing videos"):
            video_group = src[video_key]
            dst_video_group = dst.create_group(video_key)
            
            # keypoints_scaled 처리 (int16으로 변환)
            if 'keypoints_scaled' in video_group:
                scaled_data = video_group['keypoints_scaled'][:]
                dst_video_group.create_dataset(
                    'keypoints_scaled',
                    data=scaled_data.astype(np.int16),
                    compression='lzf',
                    shuffle=True
                )
            
            # scores 처리 (float32 유지)
            if 'scores' in video_group:
                scores_data = video_group['scores'][:]
                dst_video_group.create_dataset(
                    'scores',
                    data=scores_data,
                    compression='lzf',
                    shuffle=True
                )
            
            # keypoints_original은 스킵 (복사하지 않음)

def get_file_size_mb(filepath):
    """파일 크기를 MB 단위로 반환"""
    return os.path.getsize(filepath) / (1024 * 1024)

def process_batch_files(input_pattern, output_dir):
    """
    배치 파일들을 일괄 처리
    
    Args:
        input_pattern: 입력 파일 패턴 (예: "batch_*_*_F_poses.h5")
        output_dir: 출력 디렉토리
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 파일 목록 가져오기
    input_files = glob.glob(input_pattern)
    input_files.sort()
    
    if not input_files:
        print(f"No files found matching pattern: {input_pattern}")
        return
    
    print(f"Found {len(input_files)} files to process")
    
    total_original_size = 0
    total_optimized_size = 0
    
    for input_file in input_files:
        # 출력 파일 경로 생성
        filename = os.path.basename(input_file)
        name_parts = filename.replace('.h5', '').split('_')
        # batch_03_00_F_poses -> batch_03_00_F_poses_optimized
        output_filename = '_'.join(name_parts) + '_optimized.h5'
        output_file = os.path.join(output_dir, output_filename)
        
        # 원본 파일 크기
        original_size = get_file_size_mb(input_file)
        total_original_size += original_size
        
        try:
            # 최적화 수행
            optimize_h5_file(input_file, output_file)
            
            # 최적화된 파일 크기
            optimized_size = get_file_size_mb(output_file)
            total_optimized_size += optimized_size
            
            # 압축률 계산
            compression_ratio = (1 - optimized_size / original_size) * 100
            
            print(f"✓ {filename}")
            print(f"  Original: {original_size:.1f} MB")
            print(f"  Optimized: {optimized_size:.1f} MB")
            print(f"  Compression: {compression_ratio:.1f}%")
            print()
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            continue
    
    # 전체 통계
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total files processed: {len(input_files)}")
    print(f"Total original size: {total_original_size:.1f} MB")
    print(f"Total optimized size: {total_optimized_size:.1f} MB")
    print(f"Total space saved: {total_original_size - total_optimized_size:.1f} MB")
    print(f"Overall compression: {(1 - total_optimized_size / total_original_size) * 100:.1f}%")

def verify_optimized_file(original_file, optimized_file):
    """
    최적화된 파일이 올바르게 생성되었는지 검증
    """
    print(f"Verifying: {optimized_file}")
    
    with h5py.File(original_file, 'r') as orig, h5py.File(optimized_file, 'r') as opt:
        # 비디오 수 확인
        orig_videos = list(orig.keys())
        opt_videos = list(opt.keys())
        
        if len(orig_videos) != len(opt_videos):
            print(f"✗ Video count mismatch: {len(orig_videos)} vs {len(opt_videos)}")
            return False
        
        # 샘플 비디오 확인
        sample_video = orig_videos[0]
        orig_group = orig[sample_video]
        opt_group = opt[sample_video]
        
        # keypoints_original이 삭제되었는지 확인
        if 'keypoints_original' in opt_group:
            print("✗ keypoints_original should be deleted")
            return False
        
        # keypoints_scaled 데이터 타입 확인
        if 'keypoints_scaled' in opt_group:
            if opt_group['keypoints_scaled'].dtype != np.int16:
                print(f"✗ keypoints_scaled dtype should be int16, got {opt_group['keypoints_scaled'].dtype}")
                return False
        
        # scores 데이터 타입 확인
        if 'scores' in opt_group:
            if opt_group['scores'].dtype != np.float32:
                print(f"✗ scores dtype should be float32, got {opt_group['scores'].dtype}")
                return False
        
        # 데이터 shape 확인
        if 'keypoints_scaled' in orig_group and 'keypoints_scaled' in opt_group:
            orig_shape = orig_group['keypoints_scaled'].shape
            opt_shape = opt_group['keypoints_scaled'].shape
            if orig_shape != opt_shape:
                print(f"✗ Shape mismatch: {orig_shape} vs {opt_shape}")
                return False
        
        print("✓ Verification passed")
        return True

# 사용 예제
if __name__ == "__main__":
    # 현재 디렉토리의 모든 배치 파일 처리
    input_pattern = "batch_*_*_F_poses.h5"
    output_directory = "optimized_h5_files"
    
    # 일괄 처리 실행
    process_batch_files(input_pattern, output_directory)
    
    # 첫 번째 파일 검증 (옵션)
    input_files = glob.glob(input_pattern)
    if input_files:
        first_file = input_files[0]
        filename = os.path.basename(first_file)
        name_parts = filename.replace('.h5', '').split('_')
        output_filename = '_'.join(name_parts) + '_optimized.h5'
        output_file = os.path.join(output_directory, output_filename)
        
        if os.path.exists(output_file):
            verify_optimized_file(first_file, output_file)