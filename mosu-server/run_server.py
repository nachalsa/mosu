#!/usr/bin/env python3
"""
MOSU 서버 실행 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 메인 서버 실행
if __name__ == "__main__":
    from main import main
    main()
