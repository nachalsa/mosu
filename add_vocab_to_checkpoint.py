#!/usr/bin/env python3
"""
기존 체크포인트에 vocabulary 정보를 추가하는 스크립트
"""

import torch
import json
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_vocab_to_checkpoint(checkpoint_path: str, output_path: str = None):
    """체크포인트에 vocabulary 정보 추가"""
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    if output_path is None:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_with_vocab.pt"
    else:
        output_path = Path(output_path)
    
    logger.info(f"📂 체크포인트 로딩: {checkpoint_path}")
    
    # 기존 체크포인트 로드 (weights_only=False로 안전하게 로드)
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info("✅ 체크포인트 로드 성공")
    except Exception as e:
        logger.error(f"❌ 체크포인트 로드 실패: {e}")
        return False
    
    # 기본 한국어 수화 vocabulary (일반적으로 사용되는 단어들)
    default_vocab = [
        # 기본 인사
        "안녕하세요", "안녕히가세요", "안녕히계세요", "만나서반가워요",
        "감사합니다", "고맙습니다", "죄송합니다", "미안합니다",
        
        # 기본 응답
        "네", "아니요", "좋아요", "싫어요", "괜찮아요", "모르겠어요",
        
        # 감정 표현
        "기쁘다", "슬프다", "화나다", "무섭다", "행복하다", "걱정하다",
        "사랑해요", "좋아해요", "미워해요",
        
        # 동작 동사
        "가다", "오다", "앉다", "서다", "걷다", "뛰다",
        "먹다", "마시다", "보다", "듣다", "말하다", "웃다", "울다", "자다",
        "읽다", "쓰다", "그리다", "노래하다", "춤추다",
        
        # 장소
        "집", "학교", "회사", "병원", "상점", "식당", "공원", "도서관",
        "화장실", "방", "부엌", "거실",
        
        # 사람
        "엄마", "아빠", "아들", "딸", "형", "누나", "동생", "할머니", "할아버지",
        "친구", "선생님", "의사", "간호사", "경찰", "소방관",
        
        # 사물
        "물", "밥", "빵", "과일", "책", "연필", "컴퓨터", "전화", "시계",
        "옷", "신발", "가방", "돈", "차", "버스", "지하철",
        
        # 시간
        "오늘", "어제", "내일", "아침", "점심", "저녁", "밤",
        "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일",
        
        # 숫자 (기본)
        "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열",
        
        # 질문 단어
        "누구", "언제", "어디", "무엇", "왜", "어떻게",
        
        # 기타 유용한 단어들
        "있다", "없다", "많다", "적다", "크다", "작다", "길다", "짧다",
        "뜨겁다", "차갑다", "달다", "쓰다", "맵다",
        "빨갛다", "파랗다", "노랗다", "검정", "하양",
        "도와주세요", "천천히", "빨리", "조용히", "크게"
    ]
    
    # 기존에 vocabulary가 있는지 확인
    if 'vocab_words' in checkpoint and checkpoint['vocab_words']:
        logger.info(f"✅ 기존 vocabulary 발견: {len(checkpoint['vocab_words'])}개 단어")
        vocab_words = checkpoint['vocab_words']
    else:
        logger.info(f"⚠️ 기존 vocabulary 없음, 기본 vocabulary 사용: {len(default_vocab)}개 단어")
        vocab_words = default_vocab
    
    # word_to_id 매핑 생성
    word_to_id = {word: i for i, word in enumerate(vocab_words)}
    
    # vocabulary 정보 추가
    checkpoint['vocab_words'] = vocab_words
    checkpoint['word_to_id'] = word_to_id
    checkpoint['vocab_size'] = len(vocab_words)
    
    # 모델 설정 정보 추가 (없으면)
    if 'model_config' not in checkpoint:
        checkpoint['model_config'] = {
            'vocab_size': len(vocab_words),
            'embed_dim': 256,  # 기본값
            'num_encoder_layers': 6,
            'num_decoder_layers': 4,
            'num_heads': 8,
            'dim_feedforward': 1024,
            'max_seq_len': 200,
            'dropout': 0.1
        }
        logger.info("✅ 모델 설정 정보 추가")
    
    # 기존 정보 로그
    existing_keys = list(checkpoint.keys())
    logger.info(f"📋 체크포인트 키: {existing_keys}")
    
    # 새로운 체크포인트 저장
    logger.info(f"💾 새 체크포인트 저장: {output_path}")
    
    try:
        torch.save(checkpoint, output_path)
        logger.info("✅ 저장 완료!")
        
        # 검증: 저장된 파일 다시 로드해서 확인
        logger.info("🔍 저장된 파일 검증 중...")
        verification_checkpoint = torch.load(output_path, map_location='cpu', weights_only=False)
        
        if 'vocab_words' in verification_checkpoint:
            logger.info(f"✅ 검증 성공: {len(verification_checkpoint['vocab_words'])}개 단어 포함")
            
            # 샘플 단어 출력
            sample_words = verification_checkpoint['vocab_words'][:10]
            logger.info(f"📝 샘플 단어: {sample_words}")
        else:
            logger.error("❌ 검증 실패: vocabulary 정보 없음")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 저장 실패: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="체크포인트에 vocabulary 추가")
    parser.add_argument("checkpoint", help="입력 체크포인트 파일 경로")
    parser.add_argument("--output", help="출력 체크포인트 파일 경로")
    parser.add_argument("--overwrite", action="store_true", help="원본 파일 덮어쓰기")
    
    args = parser.parse_args()
    
    output_path = args.output
    if args.overwrite:
        output_path = args.checkpoint
    
    success = add_vocab_to_checkpoint(args.checkpoint, output_path)
    
    if success:
        logger.info("🎉 작업 완료!")
    else:
        logger.error("❌ 작업 실패")
        exit(1)

if __name__ == "__main__":
    main()
