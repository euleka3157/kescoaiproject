# 프로젝트 환경 설정

## Python 환경
- **Python 경로**: `/c/Users/user/AppData/Local/Programs/Python/Python313/python.exe`
- **Python 버전**: 3.13
- **대체 경로**: `/c/Users/user/AppData/Local/Programs/Python/Python310/python.exe` (3.10)

## 실행 방법
```bash
# UTF-8 인코딩으로 실행 (한글 출력 지원)
PYTHONIOENCODING=utf-8 /c/Users/user/AppData/Local/Programs/Python/Python313/python.exe -X utf8 "스크립트경로"

# 대화형 입력 자동화 예시
PYTHONIOENCODING=utf-8 printf '입력1\n입력2\n' | /c/Users/user/AppData/Local/Programs/Python/Python313/python.exe -X utf8 "스크립트경로"
```

## 주요 스크립트
- `building_multi_risk_analyzer.py`: 건물 노후위험도 × 홍수위험 × 산사태근접위험 통합 분석

## 데이터 폴더 구조
- `건물연령/`: 건물 SHP 파일
- `홍수위험/`: 홍수위험지역 SHP 파일
- `산사태위험/`: 산사태위험 TIF 래스터 파일
- `분석결과/`: 분석 결과 출력 폴더
