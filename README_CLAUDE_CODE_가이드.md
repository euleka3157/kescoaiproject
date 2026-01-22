# 건물연령 × 홍수위험 교차분석 - Claude Code 가이드

## 📋 개요
건물연령 SHP 파일과 홍수위험 SHP 파일을 비교분석하여 
홍수위험지역에 위치한 건물들을 찾아내는 프로그램입니다.

## 🚀 시작하기

### 1단계: 필요 라이브러리 설치
PowerShell에서 다음 명령어를 실행하세요:

```powershell
pip install geopandas shapely fiona pyproj rtree pandas numpy
```

### 2단계: 프로젝트 폴더 구조 확인
```
C:\Users\user\Downloads\kescoaitest\
├── 건물연령\
│   ├── 전북\
│   │   └── 전주시 완산구\
│   │       └── AL_D196_52111_20240214.shp
│   ├── 부산\
│   └── 서울\
├── 홍수위험\
│   ├── 전북\
│   │   └── 전주시 완산구\
│   │       └── CFM_SGG_52111_030.shp
│   ├── 부산\
│   └── 서울\
└── 분석결과\  (자동 생성됨)
```

### 3단계: 스크립트 실행
```powershell
cd C:\Users\user\Downloads\kescoaitest
python building_flood_intersection.py
```

## 🔧 Claude Code에서 사용하기

### 기본 실행 (전체 분석)
Claude Code에 다음과 같이 요청하세요:

```
building_flood_intersection.py 스크립트를 실행해서 
C:\Users\user\Downloads\kescoaitest 폴더의 
건물연령과 홍수위험 데이터를 분석해줘
```

### 단일 지역 테스트
```
전주시 완산구(지역코드 52111)의 건물연령과 홍수위험 
SHP 파일만 먼저 테스트로 분석해줘
```

## 📊 출력 결과

### 생성되는 파일
- `분석결과/위험건물_52111.shp` - 위험건물 SHP
- `분석결과/분석요약.csv` - 전체 분석 요약

### 결과 SHP 파일의 컬럼 설명
| 컬럼명 | 설명 |
|--------|------|
| 원본 컬럼들 | 건물연령 SHP의 모든 원본 컬럼 |
| `intersect_area` | 교차 면적 (㎡) |
| `overlap_pct` | 건물 대비 교차 비율 (%) |
| `is_flood_risk` | 홍수위험지역 교차 여부 |
| `risk_level` | 위험등급 (Low/Medium/High/Critical) |
| `flood_*` | 홍수위험 SHP의 컬럼들 (접두사 추가) |

### 위험등급 기준
- **Low**: 겹침 비율 0-10%
- **Medium**: 겹침 비율 10-30%
- **High**: 겹침 비율 30-50%
- **Critical**: 겹침 비율 50-100%

## 🔄 다음 단계 확장

### 추가할 레이어들
1. 산사태 위험지역 근접지역
2. 침수흔적도
3. 화재이력건물

### 확장 방법
```python
# 기존 코드에 새로운 분석 레이어 추가
class MultiRiskAnalyzer(BuildingFloodAnalyzer):
    def add_landslide_risk(self, landslide_shp):
        # 산사태 위험 분석 추가
        pass
    
    def add_flood_history(self, flood_history_shp):
        # 침수흔적 분석 추가
        pass
    
    def add_fire_history(self, fire_shp):
        # 화재이력 분석 추가
        pass
    
    def calculate_composite_risk(self):
        # 종합 위험등급 계산
        pass
```

## ⚡ 성능 최적화 포인트

1. **Spatial Index (STRtree)**: O(n²) → O(n log n) 개선
2. **Bounding Box 사전 필터링**: 불필요한 정밀 계산 제거
3. **청크 처리**: 대용량 데이터 메모리 관리
4. **병렬 처리**: 여러 지역 동시 분석 가능

## 🐛 문제 해결

### 인코딩 오류
```python
gpd.read_file(shp_file, encoding='cp949')  # 또는 'euc-kr'
```

### CRS 불일치
```python
# 자동으로 CRS를 통일합니다
flood_gdf = flood_gdf.to_crs(building_gdf.crs)
```

### 메모리 부족
```python
# 청크 단위로 처리
chunk_size = 10000
for i in range(0, len(gdf), chunk_size):
    chunk = gdf.iloc[i:i+chunk_size]
    # 처리...
```

## 📞 다음 요청 예시

Claude Code에 다음과 같이 요청할 수 있습니다:

1. "분석 결과를 지도로 시각화해줘"
2. "위험등급별 건물 수를 차트로 보여줘"
3. "산사태 위험지역 데이터도 추가해서 분석해줘"
4. "기상청 API 연동 코드를 추가해줘"
5. "전국 전기설비 목록과 매칭하는 기능을 추가해줘"
