#!/usr/bin/env python3
"""
LZF HDF5 파일 구조 확인 및 PyTorch DataLoader 테스트
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time

def test_lzf_file_structure():
    """LZF HDF5 파일 구조 확인"""
    print("🔍 LZF HDF5 파일 구조 분석")
    print("=" * 50)
    
    with h5py.File('sign_language_dataset_lzf.h5', 'r') as f:
        # 기본 정보
        print("📊 기본 정보:")
        print(f"   파일 크기: {sum(f[name].size * f[name].dtype.itemsize for name in f.keys() if isinstance(f[name], h5py.Dataset)) / 1024**2:.2f}MB")
        
        # 메타데이터
        print("\n📋 메타데이터:")
        if hasattr(f, 'attrs'):
            for key in f.attrs.keys():
                print(f"   {key}: {f.attrs[key]}")
        
        # vocabulary 정보
        print("\n📖 Vocabulary:")
        if 'vocabulary' in f:
            vocab_size = len(f['vocabulary/words'])
            print(f"   전체 단어 수: {vocab_size:,}개")
            
            # 상위 단어들
            top_words = [w.decode() if hasattr(w, 'decode') else str(w) for w in f['vocabulary/words'][:10]]
            top_freqs = f['vocabulary/frequencies'][:10] if 'frequencies' in f['vocabulary'] else []
            print(f"   최빈 단어 TOP 10: {top_words}")
            if len(top_freqs) > 0:
                print(f"   빈도수: {top_freqs.tolist()}")
        
        # 세그먼트 정보
        print("\n🎯 세그먼트 데이터:")
        if 'segments' in f:
            n_segments = len(f['segments/data_types'])
            print(f"   총 세그먼트: {n_segments:,}개")
            
            # 데이터 타입 분포
            data_types = f['segments/data_types'][:]
            word_count = np.sum(data_types == 0)
            sen_count = np.sum(data_types == 1)
            print(f"   WORD 세그먼트: {word_count:,}개")
            print(f"   SEN 세그먼트: {sen_count:,}개")
            
            # 시점 분포
            views = f['segments/views'][:]
            view_names = ['F', 'U', 'D', 'L', 'R']
            view_counts = {view_names[i]: np.sum(views == i) for i in range(5)}
            print(f"   시점 분포: {view_counts}")
            
            # vocab 길이 통계
            vocab_lens = f['segments/vocab_lens'][:]
            print(f"   평균 단어 길이: {np.mean(vocab_lens):.2f}개")
            print(f"   최대 단어 길이: {np.max(vocab_lens)}개")
            
            # 샘플 데이터
            print(f"\n   첫 10개 세그먼트 샘플:")
            for i in range(min(10, n_segments)):
                dtype_name = 'WORD' if data_types[i] == 0 else 'SEN'
                data_id = f['segments/data_ids'][i]
                vocab_len = vocab_lens[i]
                duration = f['segments/duration_frames'][i]
                print(f"     {i+1:2d}. {dtype_name}{data_id:04d} - {vocab_len}단어, {duration}프레임")


class SignLanguageDataset(Dataset):
    """수화 데이터셋 (LZF HDF5 기반)"""
    
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        
        # 데이터셋 정보 로드
        with h5py.File(hdf5_path, 'r') as f:
            self.n_segments = len(f['segments']['data_types'])
            
            # vocabulary 로드
            self.words = [w.decode() if hasattr(w, 'decode') else str(w) 
                         for w in f['vocabulary']['words'][:]]
            self.word_to_id = {word: idx for idx, word in enumerate(self.words)}
            self.vocab_size = len(self.words)
            
            # view 매핑
            self.view_names = ['F', 'U', 'D', 'L', 'R']
    
    def __len__(self):
        return self.n_segments
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            # 세그먼트 정보
            data_type = int(f['segments']['data_types'][idx])  # 0=WORD, 1=SEN
            data_id = int(f['segments']['data_ids'][idx])
            real_id = int(f['segments']['real_ids'][idx])
            view = int(f['segments']['views'][idx])
            start_frame = int(f['segments']['start_frames'][idx])
            end_frame = int(f['segments']['end_frames'][idx])
            duration = int(f['segments']['duration_frames'][idx])
            
            # vocab_ids (패딩 제거)
            vocab_len = int(f['segments']['vocab_lens'][idx])
            vocab_ids = f['segments']['vocab_ids'][idx, :vocab_len].tolist()
            
            return {
                'data_type': data_type,  # 0=WORD, 1=SEN
                'data_id': data_id,
                'real_id': real_id,
                'view': view,  # 0=F, 1=U, 2=D, 3=L, 4=R
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration': duration,
                'vocab_ids': torch.tensor(vocab_ids, dtype=torch.long),
                'vocab_len': vocab_len
            }


def test_pytorch_dataloader():
    """PyTorch DataLoader 테스트"""
    print("\n🐍 PyTorch DataLoader 테스트")
    print("=" * 50)
    
    try:
        # 데이터셋 생성
        dataset = SignLanguageDataset('sign_language_dataset_lzf.h5')
        print(f"✅ 데이터셋 로드 성공: {len(dataset):,}개 세그먼트")
        print(f"   Vocabulary 크기: {dataset.vocab_size:,}개")
        
        # DataLoader 생성 (작은 배치로 테스트)
        def collate_fn(batch):
            """배치 처리 함수"""
            # 가변 길이 vocab_ids 처리
            max_vocab_len = max(item['vocab_len'] for item in batch) if batch else 0
            
            batch_data = {}
            for key in batch[0].keys():
                if key == 'vocab_ids':
                    # 패딩 처리
                    padded_ids = []
                    for item in batch:
                        ids = item[key]
                        padded = torch.cat([ids, torch.zeros(max_vocab_len - len(ids), dtype=torch.long)])
                        padded_ids.append(padded)
                    batch_data[key] = torch.stack(padded_ids)
                else:
                    batch_data[key] = torch.tensor([item[key] for item in batch])
            
            return batch_data
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        
        # 첫 번째 배치 테스트
        print("\n🔬 첫 번째 배치 분석:")
        batch = next(iter(dataloader))
        
        print(f"   배치 크기: {batch['data_type'].shape[0]}")
        print(f"   데이터 타입: {batch['data_type'].tolist()}")
        print(f"   평균 단어 길이: {batch['vocab_len'].float().mean():.2f}")
        print(f"   최대 단어 길이: {batch['vocab_len'].max()}")
        print(f"   vocab_ids 형태: {batch['vocab_ids'].shape}")
        
        # 개별 샘플 확인
        print(f"\n📋 첫 번째 샘플 상세:")
        sample = dataset[0]
        dtype_name = 'WORD' if sample['data_type'] == 0 else 'SEN'
        view_name = dataset.view_names[sample['view']]
        print(f"   타입: {dtype_name}{sample['data_id']:04d}")
        print(f"   시점: {view_name}")
        print(f"   프레임: {sample['start_frame']} ~ {sample['end_frame']} ({sample['duration']}프레임)")
        print(f"   단어 ID: {sample['vocab_ids'].tolist()}")
        
        # 성능 테스트
        print(f"\n⚡ 성능 테스트:")
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 10개 배치만 테스트
                break
        load_time = time.time() - start_time
        
        print(f"   10개 배치 로드 시간: {load_time:.4f}초")
        print(f"   배치당 평균 시간: {load_time/10:.4f}초")
        
        print("\n✅ PyTorch DataLoader 테스트 성공!")
        
    except Exception as e:
        print(f"❌ PyTorch DataLoader 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def performance_comparison():
    """성능 비교 (LZF vs 일반 방식)"""
    print("\n📊 성능 비교")
    print("=" * 30)
    
    try:
        # LZF 읽기 성능
        lzf_times = []
        for i in range(5):
            start = time.time()
            with h5py.File('sign_language_dataset_lzf.h5', 'r') as f:
                data = f['segments/data_types'][:1000]  # 1000개 읽기
            lzf_times.append(time.time() - start)
        
        avg_lzf_time = np.mean(lzf_times)
        
        print(f"🚀 LZF 압축:")
        print(f"   평균 읽기 시간 (1000개): {avg_lzf_time:.4f}초")
        print(f"   초당 처리량: {1000/avg_lzf_time:.0f}개/초")
        
        # 파일 크기 확인
        import os
        file_size = os.path.getsize('sign_language_dataset_lzf.h5') / (1024**2)
        print(f"   파일 크기: {file_size:.2f}MB")
        
    except Exception as e:
        print(f"성능 비교 실패: {e}")


if __name__ == "__main__":
    # 1. 파일 구조 분석
    test_lzf_file_structure()
    
    # 2. PyTorch DataLoader 테스트
    test_pytorch_dataloader()
    
    # 3. 성능 비교
    performance_comparison()
    
    print("\n" + "🎉 전체 테스트 완료!" + "\n" + "=" * 50)
