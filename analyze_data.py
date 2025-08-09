#!/usr/bin/env python3
"""
데이터 분석 스크립트 - 유효 세그먼트 수 변화 원인 파악
"""

import glob
import h5py
import numpy as np
from pathlib import Path

def analyze_pose_files():
    """포즈 파일 분석"""
    print("=== 포즈 파일 분석 ===")
    
    files = glob.glob('./data/batch_SEN_*_poses.h5')
    real_ids = set()
    
    print(f"총 포즈 파일 수: {len(files)}개")
    
    for f in files:
        parts = Path(f).name.split('_')
        if len(parts) >= 3:
            try:
                real_id = int(parts[2])
                real_ids.add(real_id)
            except:
                continue
    
    print(f"포즈 데이터가 있는 Real IDs: {sorted(real_ids)}")
    print(f"총 개수: {len(real_ids)}명")
    print(f"16명 대비 비율: {len(real_ids)/16:.3f}")
    
    return real_ids

def analyze_annotation_file():
    """어노테이션 파일 분석"""
    print("\n=== 어노테이션 파일 분석 ===")
    
    with h5py.File('./data/sign_language_dataset_only_sen_lzf.h5', 'r') as f:
        total_segments = len(f['segments']['data_ids'][:])
        print(f"총 세그먼트 수: {total_segments}개")
        
        # Real ID 분포
        real_ids = f['segments']['real_ids'][:]
        unique_real_ids = np.unique(real_ids)
        print(f"어노테이션 Real IDs: {sorted(unique_real_ids)}")
        print(f"어노테이션 Real ID 개수: {len(unique_real_ids)}명")
        
        # 데이터 타입과 뷰 분포
        data_types = f['segments']['data_types'][:]
        views = f['segments']['views'][:]
        durations = f['segments']['duration_frames'][:]
        
        sen_count = np.sum(data_types == 1)  # SEN 타입
        front_count = np.sum((data_types == 1) & (views == 0))  # SEN & 정면
        
        print(f"SEN 타입 세그먼트: {sen_count}개")
        print(f"SEN & 정면(F): {front_count}개")
        
        # 길이 필터링 (10-200)
        length_filtered = np.sum((data_types == 1) & (views == 0) & (durations >= 10) & (durations <= 200))
        print(f"SEN & 정면 & 길이조건(10-200): {length_filtered}개")
        
        return unique_real_ids, front_count, length_filtered

def analyze_intersection(pose_real_ids, annotation_real_ids, length_filtered_count):
    """교집합 분석"""
    print("\n=== 교집합 분석 ===")
    
    common_real_ids = set(pose_real_ids) & set(annotation_real_ids)
    print(f"공통 Real IDs: {sorted(common_real_ids)}")
    print(f"공통 Real ID 개수: {len(common_real_ids)}명")
    
    # 예상 유효 세그먼트 수
    expected_ratio = len(common_real_ids) / len(annotation_real_ids)
    expected_valid = int(length_filtered_count * expected_ratio)
    
    print(f"예상 유효 세그먼트 수: {expected_valid}개")
    print(f"예상 비율: {expected_ratio:.3f}")

if __name__ == "__main__":
    # 분석 실행
    pose_real_ids = analyze_pose_files()
    annotation_real_ids, front_count, length_filtered = analyze_annotation_file()
    analyze_intersection(pose_real_ids, annotation_real_ids, length_filtered)
