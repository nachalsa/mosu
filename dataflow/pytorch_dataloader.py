
# PyTorch DataLoader for LZF Sign Language Dataset
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SignLanguageDataset(Dataset):
    """수화 데이터셋 (LZF HDF5 기반)"""
    
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        
        # 데이터셋 정보 로드
        with h5py.File(hdf5_path, 'r') as f:
            self.n_segments = len(f['segments']['data_types'])
            self.vocab_size = f.attrs['vocabulary_size']
            
            # vocabulary 로드
            self.words = [w.decode() for w in f['vocabulary']['words'][:]]
            self.word_to_id = {word: idx for idx, word in enumerate(self.words)}
            
            # view 매핑
            self.view_names = ['F', 'U', 'D', 'L', 'R']
    
    def __len__(self):
        return self.n_segments
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            # 세그먼트 정보
            data_type = f['segments']['data_types'][idx]  # 0=WORD, 1=SEN
            data_id = f['segments']['data_ids'][idx]
            real_id = f['segments']['real_ids'][idx]
            view = f['segments']['views'][idx]
            start_frame = f['segments']['start_frames'][idx]
            end_frame = f['segments']['end_frames'][idx]
            duration = f['segments']['duration_frames'][idx]
            
            # vocab_ids (패딩 제거)
            vocab_len = f['segments']['vocab_lens'][idx]
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

# 사용 예시
def create_dataloader(hdf5_path: str, batch_size: int = 32, shuffle: bool = True):
    """DataLoader 생성"""
    dataset = SignLanguageDataset(hdf5_path)
    
    def collate_fn(batch):
        """배치 처리 함수"""
        # 가변 길이 vocab_ids 처리
        max_vocab_len = max(item['vocab_len'] for item in batch)
        
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
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# 사용법
if __name__ == "__main__":
    # DataLoader 생성
    dataloader = create_dataloader("sign_language_dataset_lzf.h5", batch_size=16)
    
    # 첫 번째 배치 확인
    for batch in dataloader:
        print("배치 크기:", batch['data_type'].shape[0])
        print("데이터 타입 분포:", torch.bincount(batch['data_type']))
        print("vocabulary ID 형태:", batch['vocab_ids'].shape)
        print("평균 단어 길이:", batch['vocab_len'].float().mean())
        break
