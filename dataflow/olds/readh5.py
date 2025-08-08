import h5py
import os

# Sign Language Dataset 구조 확인
print("=== Sign Language Dataset LZF 구조 ===")
try:
    with h5py.File('../mosuModel/data/sign_language_dataset_lzf.h5', 'r') as f:
        print("Keys:", list(f.keys()))
        
        # 메타데이터 확인
        print("\nMetadata:")
        for key in f.attrs.keys():
            print(f"  {key}: {f.attrs[key]}")
        
        # vocabulary 확인
        if 'vocabulary' in f:
            print(f"\nVocabulary:")
            vocab_size = len(f['vocabulary/words'])
            print(f"  Total words: {vocab_size}")
            words = [w.decode() if hasattr(w, 'decode') else str(w) for w in f['vocabulary/words'][:10]]
            print(f"  Top 10 words: {words}")
        
        # segments 확인
        if 'segments' in f:
            print(f"\nSegments:")
            n_segments = len(f['segments/data_types'])
            print(f"  Total segments: {n_segments:,}")
            
            # 첫 번째 샘플 확인
            print(f"\n  First sample:")
            print(f"    Data type: {f['segments/data_types'][0]}")
            print(f"    Data ID: {f['segments/data_ids'][0]}")
            print(f"    Real ID: {f['segments/real_ids'][0]}")
            print(f"    Start frame: {f['segments/start_frames'][0]}")
            print(f"    End frame: {f['segments/end_frames'][0]}")
            print(f"    Duration: {f['segments/duration_frames'][0]}")
            vocab_len = f["segments/vocab_lens"][0]
            print(f"    Vocab IDs: {f['segments/vocab_ids'][0, :vocab_len].tolist()}")
except Exception as e:
    print(f"Error reading sign_language_dataset_lzf.h5: {e}")

print("\n" + "="*60)
print("=== Pose Dataset 구조 ===")
try:
    with h5py.File('../mosuModel/data/poses/batch_03_00_F_poses.h5', 'r') as f:
        print("Keys:", list(f.keys()))
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: Shape={obj.shape}, Type={obj.dtype}")
            else:
                print(f"{name}/")
        
        f.visititems(print_structure)
        
        # 첫 번째 비디오 샘플 확인
        first_video = list(f.keys())[0]
        print(f"\nFirst video ({first_video}):")
        if f'{first_video}/frames' in f:
            frames_shape = f[f'{first_video}/frames'].shape
            print(f"  Frames shape: {frames_shape}")  # (n_frames, height, width, channels)
        if f'{first_video}/metadata' in f:
            metadata = f[f'{first_video}/metadata'][()]
            print(f"  Metadata type: {type(metadata)}")
            if hasattr(metadata, 'decode'):
                print(f"  Metadata: {metadata.decode()[:200]}...")
                
except Exception as e:
    print(f"Error reading poses file: {e}")