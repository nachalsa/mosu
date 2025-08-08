#!/usr/bin/env python3
"""
완전한 수화 데이터 파이프라인 - LZF 최적화 버전
원본 morpheme 데이터 → vocabulary 기반 학습용 LZF 압축 파일

데이터 흐름:
1. 원본 morpheme JSON 파일들 수집
2. vocabulary ID 매핑으로 구조화
3. 학습 최적화된 구조로 변환
4. LZF 압축으로 최종 저장

특징:
- vocabulary ID 기반 매핑 (학습 효율성)
- 우선순위 시점 선택 (F>U>D>L>R)
- LZF 압축 (34.4배 빠른 접근)
- PyTorch 호환 구조
"""

import json
import h5py
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompletePipeline:
    """완전한 수화 데이터 파이프라인"""
    
    def __init__(self, 
                 morpheme_root: str = "/home/jy/gitwork/mmpose/jy/morpheme", 
                 fps: int = 30,
                 output_dir: str = "/home/jy/gitwork/mmpose/jy"):
        """
        Args:
            morpheme_root: morpheme 데이터 루트 경로
            fps: 영상 프레임율 (30fps)
            output_dir: 출력 디렉토리
        """
        self.morpheme_root = Path(morpheme_root)
        self.fps = fps
        self.output_dir = Path(output_dir)
        
        # 우선순위 시점 (F>U>D>L>R)
        self.view_priority = ['F', 'U', 'D', 'L', 'R']
        
        # 데이터 저장 구조
        self.raw_data = {
            'word_data': {},     # {word_id: [instances]}
            'sentence_data': {}  # {sen_id: [instances]}
        }
        
        # vocabulary 구조 (빈도순)
        self.vocabulary = {}     # {word: vocab_id}
        self.vocab_list = []     # [word1, word2, ...] 빈도순
        self.word_counter = Counter()
        
        # 통계
        self.stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'files_failed': 0,
            'view_distribution': Counter(),
            'unique_words': 0,
            'total_segments': 0
        }
        
        logger.info("🚀 Complete Pipeline 초기화 완료")
        logger.info(f"   📁 원본 경로: {self.morpheme_root}")
        logger.info(f"   🎯 출력 경로: {self.output_dir}")

    def time_to_frame(self, time_seconds: float) -> int:
        """시간(초) → 프레임 번호 변환"""
        return round(time_seconds * self.fps)

    def extract_file_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """파일명 정보 추출
        
        예: NIA_SL_WORD2947_REAL02_F_morpheme.json
        → {'type': 'WORD', 'id': 2947, 'real_id': 2, 'view': 'F'}
        """
        try:
            parts = filename.replace('.json', '').split('_')
            
            # 데이터 타입과 ID 찾기
            data_type = None
            data_id = None
            real_id = None
            view = None
            
            for i, part in enumerate(parts):
                if part.startswith('WORD'):
                    data_type = 'WORD'
                    data_id = int(part[4:])
                elif part.startswith('SEN'):
                    data_type = 'SEN'
                    data_id = int(part[3:])
                elif part.startswith('REAL'):
                    real_id = int(part[4:])
                elif part in ['F', 'U', 'D', 'L', 'R']:
                    view = part
            
            if all([data_type, data_id is not None, real_id is not None, view]):
                return {
                    'type': data_type,
                    'id': data_id,
                    'real_id': real_id,
                    'view': view
                }
                
        except (ValueError, IndexError) as e:
            logger.warning(f"파일명 파싱 실패: {filename} - {e}")
        
        return None

    def find_best_view_file(self, data_type: str, data_id: int, real_id: int, folder_path: Path) -> Optional[Tuple[Path, str]]:
        """우선순위에 따라 최적의 시점 파일 찾기"""
        
        for view in self.view_priority:
            pattern = f"NIA_SL_{data_type}{data_id:04d}_REAL{real_id:02d}_{view}_morpheme.json"
            candidate_file = folder_path / pattern
            
            if candidate_file.exists():
                return candidate_file, view
        
        return None, None

    def process_morpheme_file(self, file_path: Path, file_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """단일 morpheme 파일 처리"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 메타데이터
            metadata = data.get('metaData', {})
            morpheme_segments = data.get('data', [])
            
            # 세그먼트 처리
            processed_segments = []
            for segment in morpheme_segments:
                start_time = segment.get('start', 0.0)
                end_time = segment.get('end', 0.0)
                
                # 프레임 변환
                start_frame = self.time_to_frame(start_time)
                end_frame = self.time_to_frame(end_time)
                
                # 단어 추출 및 vocabulary 업데이트
                words = []
                vocab_ids = []
                
                if 'attributes' in segment:
                    for attr in segment['attributes']:
                        word = attr.get('name', '').strip()
                        if word:
                            words.append(word)
                            self.word_counter[word] += 1
                
                processed_segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'duration_frames': end_frame - start_frame + 1,
                    'words': words,
                    'vocab_ids': []  # 나중에 vocabulary 생성 후 할당
                })
            
            return {
                'data_type': file_info['type'],
                'data_id': file_info['id'],
                'real_id': file_info['real_id'],
                'view': file_info['view'],
                'file_path': str(file_path),
                'metadata': {
                    'duration': metadata.get('duration', 0.0),
                    'duration_frames': self.time_to_frame(metadata.get('duration', 0.0)),
                    'url': metadata.get('url', ''),
                    'exported_on': metadata.get('exportedOn', '')
                },
                'segments': processed_segments,
                'total_segments': len(processed_segments)
            }
            
        except Exception as e:
            logger.error(f"파일 처리 실패: {file_path} - {e}")
            return None

    def collect_all_data(self):
        """모든 morpheme 데이터 수집 (우선순위 시점 선택)"""
        logger.info("📊 전체 데이터 수집 시작...")
        
        # WORD 데이터 수집 (01~16 폴더)
        word_dir = self.morpheme_root / "word_morpheme"
        logger.info("📚 WORD 데이터 수집...")
        
        for folder_num in range(1, 17):  # 01~16
            folder_path = word_dir / f"{folder_num:02d}"
            if not folder_path.exists():
                continue
            
            logger.info(f"   📂 폴더 {folder_num:02d} 처리...")
            
            # 파일 그룹화 (동일 WORD+REAL 조합)
            file_groups = defaultdict(list)
            
            for json_file in folder_path.glob("*.json"):
                file_info = self.extract_file_info(json_file.name)
                if file_info and file_info['type'] == 'WORD':
                    key = (file_info['id'], file_info['real_id'])
                    file_groups[key].append((json_file, file_info))
            
            # 각 그룹에서 최적 시점 선택
            for (word_id, real_id), files in file_groups.items():
                # 우선순위에 따라 정렬
                files.sort(key=lambda x: self.view_priority.index(x[1]['view']))
                best_file, best_info = files[0]
                
                processed = self.process_morpheme_file(best_file, best_info)
                if processed:
                    if word_id not in self.raw_data['word_data']:
                        self.raw_data['word_data'][word_id] = []
                    self.raw_data['word_data'][word_id].append(processed)
                    
                    self.stats['files_processed'] += 1
                    self.stats['view_distribution'][best_info['view']] += 1
                    self.stats['total_segments'] += processed['total_segments']
        
        # SEN 데이터 수집 (01~16 폴더)
        sen_dir = self.morpheme_root / "sen_morpheme"
        logger.info("📝 SEN 데이터 수집...")
        
        for folder_num in range(1, 17):  # 01~16
            folder_path = sen_dir / f"{folder_num:02d}"
            if not folder_path.exists():
                continue
            
            logger.info(f"   📂 폴더 {folder_num:02d} 처리...")
            
            # 파일 그룹화
            file_groups = defaultdict(list)
            
            for json_file in folder_path.glob("*.json"):
                file_info = self.extract_file_info(json_file.name)
                if file_info and file_info['type'] == 'SEN':
                    key = (file_info['id'], file_info['real_id'])
                    file_groups[key].append((json_file, file_info))
            
            # 각 그룹에서 최적 시점 선택
            for (sen_id, real_id), files in file_groups.items():
                files.sort(key=lambda x: self.view_priority.index(x[1]['view']))
                best_file, best_info = files[0]
                
                processed = self.process_morpheme_file(best_file, best_info)
                if processed:
                    if sen_id not in self.raw_data['sentence_data']:
                        self.raw_data['sentence_data'][sen_id] = []
                    self.raw_data['sentence_data'][sen_id].append(processed)
                    
                    self.stats['files_processed'] += 1
                    self.stats['view_distribution'][best_info['view']] += 1
                    self.stats['total_segments'] += processed['total_segments']
        
        logger.info(f"✅ 데이터 수집 완료:")
        logger.info(f"   📚 WORD 항목: {len(self.raw_data['word_data'])}개")
        logger.info(f"   📝 SEN 항목: {len(self.raw_data['sentence_data'])}개")
        logger.info(f"   📊 처리된 파일: {self.stats['files_processed']}개")
        logger.info(f"   🎯 시점 분포: {dict(self.stats['view_distribution'])}")

    def build_vocabulary(self):
        """빈도순 vocabulary 구축 및 ID 할당"""
        logger.info("📖 Vocabulary 구축...")
        
        # 빈도순 정렬
        sorted_words = self.word_counter.most_common()
        
        # vocabulary 매핑 생성
        self.vocab_list = [word for word, count in sorted_words]
        self.vocabulary = {word: idx for idx, word in enumerate(self.vocab_list)}
        
        # 모든 세그먼트의 vocab_ids 업데이트
        self.update_vocab_ids()
        
        self.stats['unique_words'] = len(self.vocabulary)
        
        logger.info(f"✅ Vocabulary 구축 완료:")
        logger.info(f"   📖 고유 단어: {len(self.vocabulary)}개")
        logger.info(f"   🔝 최빈 단어 TOP 10: {self.vocab_list[:10]}")

    def update_vocab_ids(self):
        """모든 세그먼트의 vocab_ids 업데이트"""
        
        def update_data_vocab_ids(data_dict):
            for data_id, instances in data_dict.items():
                for instance in instances:
                    for segment in instance['segments']:
                        vocab_ids = []
                        for word in segment['words']:
                            if word in self.vocabulary:
                                vocab_ids.append(self.vocabulary[word])
                            else:
                                logger.warning(f"단어 '{word}'가 vocabulary에 없음")
                        segment['vocab_ids'] = vocab_ids
        
        update_data_vocab_ids(self.raw_data['word_data'])
        update_data_vocab_ids(self.raw_data['sentence_data'])

    def create_learning_structure(self) -> Dict[str, Any]:
        """학습 최적화된 데이터 구조 생성"""
        logger.info("🎯 학습용 데이터 구조 생성...")
        
        learning_data = {
            'metadata': {
                'version': '1.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fps': self.fps,
                'vocabulary_size': len(self.vocabulary),
                'compression': 'lzf'
            },
            'vocabulary': {
                'words': self.vocab_list,
                'word_to_id': self.vocabulary,
                'id_to_word': {idx: word for word, idx in self.vocabulary.items()},
                'word_frequencies': dict(self.word_counter.most_common())
            },
            'statistics': self.stats
        }
        
        # 학습용 segments 구조
        all_segments = []
        
        # WORD 데이터 변환
        for word_id, instances in self.raw_data['word_data'].items():
            for instance in instances:
                for segment in instance['segments']:
                    all_segments.append({
                        'data_type': 'WORD',
                        'data_id': word_id,
                        'real_id': instance['real_id'],
                        'view': instance['view'],
                        'start_frame': segment['start_frame'],
                        'end_frame': segment['end_frame'],
                        'duration_frames': segment['duration_frames'],
                        'vocab_ids': segment['vocab_ids'],
                        'words': segment['words']  # 디버깅용
                    })
        
        # SEN 데이터 변환
        for sen_id, instances in self.raw_data['sentence_data'].items():
            for instance in instances:
                for segment in instance['segments']:
                    all_segments.append({
                        'data_type': 'SEN',
                        'data_id': sen_id,
                        'real_id': instance['real_id'],
                        'view': instance['view'],
                        'start_frame': segment['start_frame'],
                        'end_frame': segment['end_frame'],
                        'duration_frames': segment['duration_frames'],
                        'vocab_ids': segment['vocab_ids'],
                        'words': segment['words']  # 디버깅용
                    })
        
        learning_data['segments'] = all_segments
        
        logger.info(f"✅ 학습용 구조 생성 완료:")
        logger.info(f"   📊 총 세그먼트: {len(all_segments)}개")
        
        return learning_data

    def save_as_lzf_hdf5(self, learning_data: Dict[str, Any]) -> str:
        """LZF 압축 HDF5 형태로 저장 (최고 성능)"""
        output_file = self.output_dir / "sign_language_dataset_lzf.h5"
        
        logger.info("💾 LZF 압축 HDF5 저장...")
        
        start_time = time.time()
        
        with h5py.File(output_file, 'w') as f:
            # 메타데이터 그룹
            meta_group = f.create_group('metadata')
            for key, value in learning_data['metadata'].items():
                if isinstance(value, str):
                    meta_group.attrs[key] = value
                else:
                    meta_group.attrs[key] = value
            
            # vocabulary 그룹 (LZF 압축)
            vocab_group = f.create_group('vocabulary')
            
            # 단어 리스트 (문자열 배열)
            words_array = np.array(learning_data['vocabulary']['words'], dtype=h5py.string_dtype())
            vocab_group.create_dataset('words', data=words_array, compression='lzf')
            
            # 단어 빈도 (정수 배열)
            frequencies = np.array([learning_data['vocabulary']['word_frequencies'][word] 
                                  for word in learning_data['vocabulary']['words']], dtype=np.int32)
            vocab_group.create_dataset('frequencies', data=frequencies, compression='lzf')
            
            # segments 데이터 (LZF 압축)
            segments_group = f.create_group('segments')
            
            segments = learning_data['segments']
            n_segments = len(segments)
            
            # 효율적인 배열 구조
            data_types = np.array([0 if s['data_type'] == 'WORD' else 1 for s in segments], dtype=np.int8)
            data_ids = np.array([s['data_id'] for s in segments], dtype=np.int16)
            real_ids = np.array([s['real_id'] for s in segments], dtype=np.int8)
            views = np.array([self.view_priority.index(s['view']) for s in segments], dtype=np.int8)
            start_frames = np.array([s['start_frame'] for s in segments], dtype=np.int32)
            end_frames = np.array([s['end_frame'] for s in segments], dtype=np.int32)
            duration_frames = np.array([s['duration_frames'] for s in segments], dtype=np.int32)
            
            # vocab_ids는 가변 길이이므로 특별 처리
            max_vocab_len = max(len(s['vocab_ids']) for s in segments) if segments else 0
            vocab_ids_padded = np.full((n_segments, max_vocab_len), -1, dtype=np.int16)  # -1로 패딩
            vocab_lens = np.zeros(n_segments, dtype=np.int8)
            
            for i, segment in enumerate(segments):
                vocab_ids = segment['vocab_ids']
                vocab_lens[i] = len(vocab_ids)
                if vocab_ids:
                    vocab_ids_padded[i, :len(vocab_ids)] = vocab_ids
            
            # LZF 압축으로 저장
            segments_group.create_dataset('data_types', data=data_types, compression='lzf')
            segments_group.create_dataset('data_ids', data=data_ids, compression='lzf')
            segments_group.create_dataset('real_ids', data=real_ids, compression='lzf')
            segments_group.create_dataset('views', data=views, compression='lzf')
            segments_group.create_dataset('start_frames', data=start_frames, compression='lzf')
            segments_group.create_dataset('end_frames', data=end_frames, compression='lzf')
            segments_group.create_dataset('duration_frames', data=duration_frames, compression='lzf')
            segments_group.create_dataset('vocab_ids', data=vocab_ids_padded, compression='lzf')
            segments_group.create_dataset('vocab_lens', data=vocab_lens, compression='lzf')
            
            # 통계 정보
            stats_group = f.create_group('statistics')
            for key, value in learning_data['statistics'].items():
                if isinstance(value, dict):
                    # Counter 객체 등 딕셔너리 처리
                    subgroup = stats_group.create_group(key)
                    for subkey, subvalue in value.items():
                        subgroup.attrs[str(subkey)] = subvalue
                else:
                    stats_group.attrs[key] = value
        
        save_time = time.time() - start_time
        file_size = output_file.stat().st_size / (1024*1024)
        
        logger.info(f"✅ LZF HDF5 저장 완료:")
        logger.info(f"   📁 파일: {output_file}")
        logger.info(f"   📊 크기: {file_size:.2f}MB")
        logger.info(f"   ⏱️  저장 시간: {save_time:.2f}초")
        
        return str(output_file)

    def create_pytorch_dataloader_code(self) -> str:
        """PyTorch DataLoader 코드 생성"""
        dataloader_code = '''
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
'''
        
        # 코드 파일로 저장
        code_file = self.output_dir / "pytorch_dataloader.py"
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(dataloader_code)
        
        logger.info(f"📝 PyTorch DataLoader 코드 생성: {code_file}")
        
        return str(code_file)

    def run_complete_pipeline(self):
        """완전한 파이프라인 실행"""
        logger.info("🚀 완전한 데이터 파이프라인 시작")
        logger.info("=" * 60)
        
        try:
            # 1단계: 데이터 수집
            self.collect_all_data()
            
            # 2단계: Vocabulary 구축
            self.build_vocabulary()
            
            # 3단계: 학습용 구조 생성
            learning_data = self.create_learning_structure()
            
            # 4단계: LZF HDF5 저장
            lzf_file = self.save_as_lzf_hdf5(learning_data)
            
            # 5단계: PyTorch DataLoader 코드 생성
            dataloader_file = self.create_pytorch_dataloader_code()
            
            # 최종 리포트
            logger.info("\n" + "🎉 파이프라인 완료!" + "\n" + "=" * 60)
            logger.info(f"📊 최종 통계:")
            logger.info(f"   📚 WORD 데이터: {len(self.raw_data['word_data'])}개")
            logger.info(f"   📝 SEN 데이터: {len(self.raw_data['sentence_data'])}개")
            logger.info(f"   📖 고유 단어: {len(self.vocabulary)}개")
            logger.info(f"   🎯 총 세그먼트: {self.stats['total_segments']}개")
            logger.info(f"   📁 LZF 파일: {lzf_file}")
            logger.info(f"   🐍 PyTorch 코드: {dataloader_file}")
            
            return {
                'lzf_file': lzf_file,
                'dataloader_file': dataloader_file,
                'statistics': self.stats,
                'vocabulary_size': len(self.vocabulary)
            }
            
        except Exception as e:
            logger.error(f"파이프라인 실패: {e}")
            raise


def main():
    """메인 실행"""
    pipeline = CompletePipeline()
    result = pipeline.run_complete_pipeline()
    
    print("\n🎯 파이프라인 완료!")
    print(f"LZF 파일: {result['lzf_file']}")
    print(f"DataLoader 코드: {result['dataloader_file']}")


if __name__ == "__main__":
    main()
