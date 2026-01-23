"""
건물 노후위험도 × 홍수위험 × 산사태근접위험 통합 분석 프로그램
================================================================
목적: 건물연령(A23 착공일자 기반) + 홍수위험 + 산사태근접위험을 결합하여
      종합 위험등급을 산출하고 필터링된 결과를 SHP로 출력

기능:
- A23(착공일자)에서 당해연도를 빼서 건물연령 계산
- 노후위험도 등급: 0-5, 5-10, 10-20, 20-30, 30-50, 50-100년
- 홍수위험: 홍수위험지역과 건물 교차 분석
- 산사태근접위험: 산사태위험지역(래스터)과 건물 간 최단거리 분석
- 종합위험등급 산출 (노후 + 홍수 + 산사태)
- 사용자 입력: 지역 선택, 연령 필터링
- cp949 인코딩 지원 (한국 SHP 파일)

사용법:
    python building_multi_risk_analyzer.py

필요 라이브러리:
    pip install geopandas shapely fiona pyproj rtree pandas numpy rasterio scipy
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.strtree import STRtree
    from shapely.geometry import shape, mapping, box
    from shapely.ops import unary_union
    import numpy as np
except ImportError as e:
    print(f"필요한 라이브러리가 없습니다: {e}")
    print("다음 명령어로 설치해주세요:")
    print("pip install geopandas shapely fiona pyproj rtree pandas numpy")
    sys.exit(1)

# 래스터 처리용 (산사태 분석)
try:
    import rasterio
    from rasterio.features import shapes
    from scipy.ndimage import distance_transform_edt
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("경고: rasterio/scipy 미설치 - 산사태 분석 기능 제한됨")
    print("설치: pip install rasterio scipy")


class BuildingMultiRiskAnalyzer:
    """건물 노후위험도 × 홍수위험 × 산사태근접위험 통합 분석 클래스"""

    # 노후위험도 등급 기준 (연령 범위, 등급명, 점수)
    AGE_RISK_LEVELS = [
        (0, 5, '신축', 1),
        (5, 10, '양호', 2),
        (10, 20, '보통', 3),
        (20, 30, '주의', 4),
        (30, 50, '노후', 5),
        (50, 100, '고령', 6),
        (100, 999, '초고령', 7)
    ]

    # 홍수위험 등급 기준 (겹침비율, 등급명, 점수)
    FLOOD_RISK_LEVELS = [
        (0, 0.01, '없음', 0),      # 교차 없음
        (0.01, 10, '낮음', 1),
        (10, 30, '보통', 2),
        (30, 50, '높음', 3),
        (50, 100, '매우높음', 4)
    ]

    # 산사태 근접위험 등급 기준 (거리(m), 등급명, 점수)
    LANDSLIDE_PROXIMITY_LEVELS = [
        (0, 50, '매우위험', 4),      # 50m 이내
        (50, 100, '위험', 3),        # 50~100m
        (100, 200, '주의', 2),       # 100~200m
        (200, 500, '관심', 1),       # 200~500m
        (500, 99999, '안전', 0)      # 500m 이상
    ]

    # 종합위험등급 기준 (점수 합계) - 3개 위험요소 합산
    COMBINED_RISK_LEVELS = [
        (0, 4, '안전', 'A'),
        (4, 6, '관심', 'B'),
        (6, 9, '주의', 'C'),
        (9, 12, '경고', 'D'),
        (12, 20, '위험', 'E')
    ]

    def __init__(self, base_path: str):
        """
        Args:
            base_path: 데이터 기본 경로
        """
        self.base_path = Path(base_path)
        self.building_path = self.base_path / "건물연령"
        self.flood_path = self.base_path / "홍수위험"
        self.landslide_path = self.base_path / "산사태위험"
        self.output_path = self.base_path / "분석결과"
        self.current_year = datetime.now().year

        # 출력 폴더 생성
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 지역 매핑 정보
        self.region_mapping = self._build_region_mapping()

        # 산사태 위험지역 캐시 (래스터→벡터 변환 결과)
        self._landslide_cache = {}

        # 시도코드 → 지역폴더명 매핑
        self.sido_to_region = {
            '11': '서울',
            '26': '부산',
            '27': '대구',
            '28': '인천',
            '29': '광주',
            '30': '대전',
            '31': '울산',
            '36': '세종',
            '41': '경기',
            '42': '강원',
            '43': '충북',
            '44': '충남',
            '45': '전북',
            '46': '전남',
            '47': '경북',
            '48': '경남',
            '50': '제주',
            '52': '전북',  # 전북 특별자치도 (신규 코드)
        }

    def _build_region_mapping(self) -> dict:
        """지역 폴더 구조를 스캔하여 매핑 정보 생성"""
        mapping = {}

        if not self.building_path.exists():
            return mapping

        for region_folder in self.building_path.iterdir():
            if region_folder.is_dir():
                region_name = region_folder.name  # 서울, 부산, 전북 등
                mapping[region_name] = {
                    'path': region_folder,
                    'districts': {}
                }

                for district_folder in region_folder.iterdir():
                    if district_folder.is_dir():
                        district_name = district_folder.name  # 종로구, 전주시 완산구 등
                        shp_files = list(district_folder.glob("*.shp"))
                        if shp_files:
                            # 지역코드 추출
                            region_code = self._get_region_code(shp_files[0].name)
                            mapping[region_name]['districts'][district_name] = {
                                'path': district_folder,
                                'code': region_code,
                                'building_shp': shp_files[0]
                            }

        return mapping

    def _get_region_code(self, filename: str) -> str:
        """파일명에서 5자리 지역코드 추출"""
        parts = filename.replace('.shp', '').split('_')
        for part in parts:
            if part.isdigit() and len(part) == 5:
                return part
        return None

    def _get_sido_code(self, region_code: str) -> str:
        """5자리 지역코드에서 2자리 시도코드 추출 (52111 → 52)"""
        if region_code and len(region_code) >= 2:
            return region_code[:2]
        return None

    def _read_shp_with_encoding(self, shp_path: Path, encodings=['cp949', 'euc-kr', 'utf-8']) -> gpd.GeoDataFrame:
        """여러 인코딩을 시도하여 SHP 파일 읽기"""
        for encoding in encodings:
            try:
                gdf = gpd.read_file(shp_path, encoding=encoding)
                return gdf
            except Exception:
                continue

        # 마지막으로 인코딩 없이 시도
        return gpd.read_file(shp_path)

    def _save_shp_with_encoding(self, gdf: gpd.GeoDataFrame, output_path: Path, encoding='cp949'):
        """cp949 인코딩으로 SHP 파일 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            gdf.to_file(output_path, encoding=encoding)
        except Exception:
            # cp949 실패시 utf-8로 저장
            gdf.to_file(output_path, encoding='utf-8')

        # cpg 파일 생성 (인코딩 명시)
        cpg_path = output_path.with_suffix('.cpg')
        with open(cpg_path, 'w') as f:
            f.write(encoding.upper())

    # =========================================================================
    # 산사태 위험지역 근접 분석 (핵심 추가 기능)
    # =========================================================================

    def find_landslide_tif(self, region_code: str) -> Path:
        """시도코드에 해당하는 산사태위험지도 TIF 파일 찾기

        폴더 구조: 산사태위험/전북/52.tif
        """
        sido_code = self._get_sido_code(region_code)

        if not sido_code:
            return None

        # 시도코드로 지역 폴더명 조회
        region_folder_name = self.sido_to_region.get(sido_code)

        if region_folder_name:
            # 지역 폴더 내에서 TIF 파일 검색 (예: 산사태위험/전북/52.tif)
            region_folder = self.landslide_path / region_folder_name

            if region_folder.exists():
                # 파일명 패턴: {시도코드}.tif (예: 52.tif)
                possible_names = [
                    f"{sido_code}.tif",
                    f"{sido_code}_tif.tif",
                    f"산사태위험지도_{sido_code}.tif"
                ]

                for pattern in possible_names:
                    tif_path = region_folder / pattern
                    if tif_path.exists():
                        return tif_path

                # 폴더 내 모든 TIF 파일 검색
                for tif_file in region_folder.glob("*.tif"):
                    if sido_code in tif_file.name:
                        return tif_file

        # 기존 방식 (하위 호환성): 루트 폴더에서 직접 검색
        possible_names = [
            f"{sido_code}.tif",
            f"{sido_code}_tif.tif",
            f"산사태위험지도_{sido_code}.tif"
        ]

        for pattern in possible_names:
            tif_path = self.landslide_path / pattern
            if tif_path.exists():
                return tif_path

        # 전체 하위 폴더 재귀 검색
        if self.landslide_path.exists():
            for tif_file in self.landslide_path.rglob("*.tif"):
                if sido_code in tif_file.name:
                    return tif_file

        return None

    def extract_landslide_risk_boundary(self, tif_path: Path, 
                                         risk_levels: list = [1, 2],
                                         simplify_tolerance: float = 20.0) -> gpd.GeoDataFrame:
        """
        산사태 위험지도 래스터에서 위험지역 경계 추출
        
        Args:
            tif_path: 산사태위험지도 TIF 파일 경로
            risk_levels: 추출할 위험등급 (1=매우위험, 2=위험)
            simplify_tolerance: 폴리곤 단순화 허용오차 (m)
            
        Returns:
            위험지역 경계 GeoDataFrame
        """
        if not RASTERIO_AVAILABLE:
            print("  rasterio 미설치로 산사태 분석 불가")
            return None
            
        # 캐시 확인
        cache_key = str(tif_path)
        if cache_key in self._landslide_cache:
            print("  산사태 위험지역 경계 (캐시 사용)")
            return self._landslide_cache[cache_key]
            
        print(f"  산사태 위험지도 로딩: {tif_path.name}")
        
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            
            print(f"    래스터 크기: {data.shape[1]} x {data.shape[0]} 픽셀")
            print(f"    추출 대상 등급: {risk_levels}")
            
            # 위험등급만 마스크 생성
            risk_mask = np.isin(data, risk_levels)
            pixel_count = np.sum(risk_mask)
            print(f"    위험지역 픽셀 수: {pixel_count:,}개")
            
            if pixel_count == 0:
                print("    위험지역 없음")
                return None
            
            # 래스터→벡터 변환 (위험지역만)
            print("    벡터 변환 중 (시간 소요)...")
            
            # 마스크를 정수형으로 변환 (shapes 함수 요구사항)
            risk_data = np.where(risk_mask, 1, 0).astype(np.int16)
            
            polygons = []
            for geom, value in shapes(risk_data, mask=risk_mask, transform=transform):
                if value == 1:
                    polygons.append(shape(geom))
            
            print(f"    추출된 폴리곤 수: {len(polygons):,}개")
            
            if not polygons:
                return None
            
            # 폴리곤 병합 (Dissolve)
            print("    폴리곤 병합 중...")
            merged = unary_union(polygons)
            
            # 단순화 (Simplify)
            print(f"    폴리곤 단순화 중 (tolerance={simplify_tolerance}m)...")
            simplified = merged.simplify(simplify_tolerance, preserve_topology=True)
            
            # GeoDataFrame 생성
            gdf = gpd.GeoDataFrame(
                {'risk_level': ['high'], 'geometry': [simplified]},
                crs=crs
            )
            
            # 캐시 저장
            self._landslide_cache[cache_key] = gdf
            
            print(f"    최종 위험지역 폴리곤: {len(gdf)}개")
            
            return gdf

    def calculate_landslide_proximity(self, building_gdf: gpd.GeoDataFrame,
                                       tif_path: Path = None) -> gpd.GeoDataFrame:
        """
        각 건물에서 산사태 위험지역까지의 최단거리 계산 (Distance Transform 방식)

        Args:
            building_gdf: 건물 GeoDataFrame
            tif_path: 산사태위험지도 TIF 파일 경로

        Returns:
            거리 정보가 추가된 건물 GeoDataFrame
        """
        return self._landslide_distance_transform(building_gdf, tif_path)

    def _landslide_distance_transform(self, building_gdf: gpd.GeoDataFrame,
                                       tif_path: Path) -> gpd.GeoDataFrame:
        """
        거리 래스터 변환 방식 (Distance Transform)
        - scipy의 distance_transform_edt로 거리 래스터 생성
        - 건물 좌표에서 거리값 샘플링 (O(1) 조회)
        """
        building_gdf = building_gdf.copy()

        if tif_path is None or not tif_path.exists():
            print("    TIF 파일 없음 - 산사태 분석 불가")
            building_gdf['산사태거리'] = 99999
            building_gdf['산사태등급'] = '분석불가'
            building_gdf['산사태점수'] = 0
            return building_gdf

        print(f"    거리 래스터 생성 중: {tif_path.name}")

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            pixel_size = abs(transform[0])  # 픽셀 크기 (m)

            print(f"      래스터 크기: {data.shape[1]} x {data.shape[0]} 픽셀")
            print(f"      픽셀 해상도: {pixel_size:.1f}m")

            # 위험지역 마스크 (등급 1, 2를 위험지역으로)
            risk_mask = np.isin(data, [1, 2])
            risk_pixel_count = np.sum(risk_mask)
            print(f"      위험지역 픽셀: {risk_pixel_count:,}개")

            if risk_pixel_count == 0:
                print("      위험지역 없음")
                building_gdf['산사태거리'] = 99999
                building_gdf['산사태등급'] = '안전'
                building_gdf['산사태점수'] = 0
                return building_gdf

            # Distance Transform: 비위험지역 픽셀에서 가장 가까운 위험지역까지 거리
            # ~risk_mask: 위험지역이 아닌 곳 = True
            print("      거리 변환 계산 중...")
            distance_pixels = distance_transform_edt(~risk_mask)
            distance_meters = distance_pixels * pixel_size

            print(f"      거리 변환 완료 (최대 거리: {distance_meters.max():.0f}m)")

            # 건물 좌표를 래스터 좌표로 변환하여 거리값 샘플링
            print(f"      건물 {len(building_gdf)}개 거리 샘플링 중...")

            # CRS 통일
            if building_gdf.crs != crs:
                building_gdf_proj = building_gdf.to_crs(crs)
            else:
                building_gdf_proj = building_gdf

            distances = []
            for idx, row in building_gdf_proj.iterrows():
                centroid = row.geometry.centroid

                # 좌표 → 래스터 인덱스 변환
                col = int((centroid.x - transform[2]) / transform[0])
                row_idx = int((centroid.y - transform[5]) / transform[4])

                # 래스터 범위 내인지 확인
                if 0 <= row_idx < data.shape[0] and 0 <= col < data.shape[1]:
                    dist = distance_meters[row_idx, col]
                else:
                    dist = 99999  # 래스터 범위 밖

                distances.append(dist)

            building_gdf['산사태거리'] = distances

        # 거리 기반 위험등급 분류
        self._classify_landslide_risk(building_gdf)

        return building_gdf

    def _classify_landslide_risk(self, gdf: gpd.GeoDataFrame) -> None:
        """산사태 거리 기반 위험등급 분류 (in-place)"""
        def classify_proximity(dist):
            for min_dist, max_dist, level_name, score in self.LANDSLIDE_PROXIMITY_LEVELS:
                if min_dist <= dist < max_dist:
                    return (level_name, score)
            return ('안전', 0)

        proximity_risk = gdf['산사태거리'].apply(classify_proximity)
        gdf['산사태등급'] = proximity_risk.apply(lambda x: x[0])
        gdf['산사태점수'] = proximity_risk.apply(lambda x: x[1])

        # 통계 출력
        risk_counts = gdf['산사태등급'].value_counts()
        print(f"      산사태 근접위험 분포:")
        for level, count in risk_counts.items():
            print(f"        {level}: {count}개")

    # =========================================================================
    # 기존 기능 (건물연령, 홍수위험)
    # =========================================================================

    def calculate_building_age(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """A23(착공일자)에서 건물연령 계산"""
        gdf = gdf.copy()

        def extract_year(date_val):
            """날짜에서 연도 추출"""
            if pd.isna(date_val):
                return None
            try:
                if isinstance(date_val, str):
                    year_str = date_val[:4]
                    return int(year_str)
                elif hasattr(date_val, 'year'):
                    return date_val.year
                else:
                    return int(str(date_val)[:4])
            except:
                return None

        # 착공연도 추출
        if 'A23' in gdf.columns:
            gdf['건축년도'] = gdf['A23'].apply(extract_year)
        else:
            # A23 컬럼이 없으면 다른 날짜 컬럼 시도
            date_cols = [col for col in gdf.columns if '일자' in col or 'date' in col.lower()]
            if date_cols:
                gdf['건축년도'] = gdf[date_cols[0]].apply(extract_year)
            else:
                gdf['건축년도'] = None

        # 건물연령 계산 (당해연도 - 착공연도)
        gdf['건물연령'] = gdf['건축년도'].apply(
            lambda x: self.current_year - x if x and x > 1900 else None
        )

        return gdf

    def classify_age_risk(self, age: float) -> tuple:
        """건물연령에 따른 노후위험도 분류"""
        if pd.isna(age) or age is None:
            return ('미확인', 0)

        for min_age, max_age, level_name, score in self.AGE_RISK_LEVELS:
            if min_age <= age < max_age:
                return (level_name, score)

        return ('초고령', 7)

    def classify_flood_risk(self, overlap_pct: float) -> tuple:
        """홍수위험 겹침비율에 따른 등급 분류"""
        if pd.isna(overlap_pct):
            return ('없음', 0)

        for min_pct, max_pct, level_name, score in self.FLOOD_RISK_LEVELS:
            if min_pct <= overlap_pct < max_pct:
                return (level_name, score)

        return ('매우높음', 4)

    def classify_combined_risk(self, total_score: int) -> tuple:
        """종합위험등급 분류 (노후 + 홍수 + 산사태)"""
        for min_score, max_score, level_name, grade in self.COMBINED_RISK_LEVELS:
            if min_score <= total_score < max_score:
                return (level_name, grade)

        return ('위험', 'E')

    def get_age_category(self, age: float) -> str:
        """연령대 카테고리 반환"""
        if pd.isna(age) or age is None:
            return '미확인'

        if age < 5:
            return '0-5년'
        elif age < 10:
            return '5-10년'
        elif age < 20:
            return '10-20년'
        elif age < 30:
            return '20-30년'
        elif age < 50:
            return '30-50년'
        elif age < 100:
            return '50-100년'
        else:
            return '100년이상'

    def list_available_regions(self):
        """사용 가능한 지역 목록 출력"""
        print("\n" + "=" * 50)
        print("사용 가능한 지역 목록")
        print("=" * 50)

        for region_name, region_info in self.region_mapping.items():
            print(f"\n[{region_name}]")
            for district_name, district_info in region_info['districts'].items():
                code = district_info.get('code', '코드없음')
                print(f"  - {district_name} (코드: {code})")

        return self.region_mapping

    def find_flood_shp(self, region_code: str) -> Path:
        """지역코드에 해당하는 홍수위험 SHP 파일 찾기"""
        for shp_file in self.flood_path.rglob("*.shp"):
            if region_code in shp_file.name:
                return shp_file
        return None

    # =========================================================================
    # 통합 분석 (노후 + 홍수 + 산사태)
    # =========================================================================

    def analyze_region(self, region_name: str, district_name: str = None,
                       min_age: int = None, max_age: int = None,
                       include_flood: bool = True,
                       include_landslide: bool = True) -> gpd.GeoDataFrame:
        """
        특정 지역 통합 분석 수행

        Args:
            region_name: 지역명 (서울, 부산, 전북 등)
            district_name: 구/군명 (None이면 해당 지역 전체)
            min_age: 최소 건물연령 필터 (이 연령 이상만 포함)
            max_age: 최대 건물연령 필터 (이 연령 이하만 포함)
            include_flood: 홍수위험 분석 포함 여부
            include_landslide: 산사태근접위험 분석 포함 여부

        Returns:
            분석 결과 GeoDataFrame
        """
        if region_name not in self.region_mapping:
            print(f"지역을 찾을 수 없습니다: {region_name}")
            return None

        region_info = self.region_mapping[region_name]

        # 분석할 구/군 목록 결정
        if district_name:
            if district_name not in region_info['districts']:
                print(f"구/군을 찾을 수 없습니다: {district_name}")
                return None
            districts_to_analyze = {district_name: region_info['districts'][district_name]}
        else:
            districts_to_analyze = region_info['districts']

        all_results = []

        for dist_name, dist_info in districts_to_analyze.items():
            print(f"\n{'='*60}")
            print(f"분석 중: {region_name} {dist_name}")
            print(f"{'='*60}")

            result = self._analyze_single_district(
                dist_info, region_name, dist_name, min_age, max_age,
                include_flood, include_landslide
            )

            if result is not None and len(result) > 0:
                all_results.append(result)

        if all_results:
            combined_result = pd.concat(all_results, ignore_index=True)
            return gpd.GeoDataFrame(combined_result, crs=all_results[0].crs)

        return None

    def _analyze_single_district(self, dist_info: dict, region_name: str,
                                  district_name: str, min_age: int, max_age: int,
                                  include_flood: bool, include_landslide: bool) -> gpd.GeoDataFrame:
        """단일 구/군 통합 분석"""
        building_shp = dist_info['building_shp']
        region_code = dist_info['code']

        print(f"  건물연령 파일: {building_shp.name}")

        # 데이터 로드
        print("  데이터 로딩 중...")
        building_gdf = self._read_shp_with_encoding(building_shp)
        print(f"  전체 건물 수: {len(building_gdf)}")

        # 건물연령 계산
        print("  건물연령 계산 중...")
        building_gdf = self.calculate_building_age(building_gdf)

        # 연령 필터링 (분석 전)
        if min_age is not None:
            before_count = len(building_gdf)
            building_gdf = building_gdf[building_gdf['건물연령'] >= min_age]
            print(f"  연령 필터링 (>={min_age}년): {before_count} → {len(building_gdf)}개")

        if max_age is not None:
            before_count = len(building_gdf)
            building_gdf = building_gdf[building_gdf['건물연령'] <= max_age]
            print(f"  연령 필터링 (<={max_age}년): {before_count} → {len(building_gdf)}개")

        if len(building_gdf) == 0:
            print("  필터링 후 건물이 없습니다.")
            return None

        # 노후위험도 계산
        age_risk = building_gdf['건물연령'].apply(self.classify_age_risk)
        building_gdf['노후등급'] = age_risk.apply(lambda x: x[0])
        building_gdf['노후점수'] = age_risk.apply(lambda x: x[1])
        building_gdf['연령대'] = building_gdf['건물연령'].apply(self.get_age_category)

        # 홍수위험 분석
        if include_flood:
            flood_shp = self.find_flood_shp(region_code)
            if flood_shp:
                print(f"  홍수위험 파일: {flood_shp.name}")
                flood_gdf = self._read_shp_with_encoding(flood_shp)
                print(f"  홍수위험구역 수: {len(flood_gdf)}")
                print("  홍수위험 교차 분석 중...")
                building_gdf = self._analyze_flood_intersection(building_gdf, flood_gdf)
            else:
                print(f"  홍수위험 데이터 없음 (코드: {region_code})")
                building_gdf['겹침비율'] = 0
                building_gdf['홍수등급'] = '없음'
                building_gdf['홍수점수'] = 0
        else:
            building_gdf['겹침비율'] = 0
            building_gdf['홍수등급'] = '미분석'
            building_gdf['홍수점수'] = 0

        # 산사태 근접위험 분석
        if include_landslide and RASTERIO_AVAILABLE:
            landslide_tif = self.find_landslide_tif(region_code)
            if landslide_tif:
                print(f"  산사태위험지도 파일: {landslide_tif.name}")
                print("  산사태 근접위험 분석 중...")
                building_gdf = self.calculate_landslide_proximity(building_gdf, tif_path=landslide_tif)
            else:
                print(f"  산사태위험지도 데이터 없음")
                building_gdf['산사태거리'] = 99999
                building_gdf['산사태등급'] = '분석불가'
                building_gdf['산사태점수'] = 0
        else:
            building_gdf['산사태거리'] = 99999
            building_gdf['산사태등급'] = '미분석'
            building_gdf['산사태점수'] = 0

        # =========================================================================
        # 조합별 위험도 계산
        # =========================================================================
        print("  조합별 위험도 계산 중...")

        # 개별 위험점수
        building_gdf['노후점수'] = building_gdf['노후점수'].fillna(0).astype(int)
        building_gdf['홍수점수'] = building_gdf['홍수점수'].fillna(0).astype(int)
        building_gdf['산사태점수'] = building_gdf['산사태점수'].fillna(0).astype(int)

        # 2개 조합 위험점수
        building_gdf['노후홍수점수'] = building_gdf['노후점수'] + building_gdf['홍수점수']
        building_gdf['홍수산사태점수'] = building_gdf['홍수점수'] + building_gdf['산사태점수']
        building_gdf['노후산사태점수'] = building_gdf['노후점수'] + building_gdf['산사태점수']

        # 종합점수 (3개 모두)
        building_gdf['종합점수'] = (
            building_gdf['노후점수'] +
            building_gdf['홍수점수'] +
            building_gdf['산사태점수']
        )

        # 2개 조합 위험등급 분류
        def classify_dual_risk(score, max_score=11):
            """2개 조합 위험등급 (최대 7+4=11점)"""
            if score <= 2:
                return ('안전', 'A')
            elif score <= 4:
                return ('관심', 'B')
            elif score <= 6:
                return ('주의', 'C')
            elif score <= 8:
                return ('경고', 'D')
            else:
                return ('위험', 'E')

        # 노후+홍수 등급
        nh_risk = building_gdf['노후홍수점수'].apply(classify_dual_risk)
        building_gdf['노후홍수등급'] = nh_risk.apply(lambda x: x[0])
        building_gdf['노후홍수코드'] = nh_risk.apply(lambda x: x[1])

        # 홍수+산사태 등급
        hs_risk = building_gdf['홍수산사태점수'].apply(classify_dual_risk)
        building_gdf['홍수산사태등급'] = hs_risk.apply(lambda x: x[0])
        building_gdf['홍수산사태코드'] = hs_risk.apply(lambda x: x[1])

        # 노후+산사태 등급
        ns_risk = building_gdf['노후산사태점수'].apply(classify_dual_risk)
        building_gdf['노후산사태등급'] = ns_risk.apply(lambda x: x[0])
        building_gdf['노후산사태코드'] = ns_risk.apply(lambda x: x[1])

        # 종합위험등급 (3개 모두)
        combined_risk = building_gdf['종합점수'].apply(self.classify_combined_risk)
        building_gdf['종합등급'] = combined_risk.apply(lambda x: x[0])
        building_gdf['위험코드'] = combined_risk.apply(lambda x: x[1])

        # =========================================================================
        # 좌표 정보 추출 (시각화용)
        # =========================================================================
        print("  좌표 정보 추출 중...")

        # 건물 중심점 좌표 추출
        building_gdf['중심점X'] = building_gdf.geometry.centroid.x
        building_gdf['중심점Y'] = building_gdf.geometry.centroid.y

        # EPSG:5186 좌표 변환 (최종 저장 시 재계산되므로 임시값)
        # 경도/위도도 EPSG:5186 기준 X/Y 좌표로 저장
        if building_gdf.crs and building_gdf.crs.to_epsg() != 5186:
            try:
                gdf_5186 = building_gdf.to_crs(epsg=5186)
                building_gdf['경도'] = gdf_5186.geometry.centroid.x
                building_gdf['위도'] = gdf_5186.geometry.centroid.y
            except Exception as e:
                print(f"    EPSG:5186 변환 실패: {e}")
                building_gdf['경도'] = building_gdf['중심점X']
                building_gdf['위도'] = building_gdf['중심점Y']
        else:
            building_gdf['경도'] = building_gdf['중심점X']
            building_gdf['위도'] = building_gdf['중심점Y']

        # 주소 정보 추출 (원본 데이터에서)
        address_cols = ['A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']  # 주소 관련 컬럼들
        for col in address_cols:
            if col in building_gdf.columns:
                # 주소 컬럼이 있으면 '주소' 컬럼으로 통합
                if '주소' not in building_gdf.columns:
                    building_gdf['주소'] = building_gdf[col].fillna('')
                else:
                    building_gdf['주소'] = building_gdf['주소'] + ' ' + building_gdf[col].fillna('')

        # 지역 정보 추가
        building_gdf['지역'] = region_name
        building_gdf['구군'] = district_name
        building_gdf['지역코드'] = region_code

        print(f"  분석 완료: {len(building_gdf)}개 건물")

        return building_gdf

    def _analyze_flood_intersection(self, building_gdf: gpd.GeoDataFrame,
                                     flood_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """홍수위험 교차 분석 (전체 건물에 대해, 교차 여부 + 비율 계산)"""
        # CRS 통일
        if building_gdf.crs != flood_gdf.crs:
            flood_gdf = flood_gdf.to_crs(building_gdf.crs)

        # Geometry 유효성 검사
        building_gdf = building_gdf.copy()
        building_gdf['geometry'] = building_gdf['geometry'].buffer(0)
        flood_gdf = flood_gdf.copy()
        flood_gdf['geometry'] = flood_gdf['geometry'].buffer(0)

        # Spatial Index 생성
        flood_tree = STRtree(flood_gdf.geometry.values)

        # 결과 저장용
        overlap_ratios = []
        total = len(building_gdf)

        for idx, (building_idx, building_row) in enumerate(building_gdf.iterrows()):
            if idx % 1000 == 0 and idx > 0:
                print(f"    진행률: {idx}/{total} ({100*idx/total:.1f}%)")

            building_geom = building_row.geometry
            candidate_indices = flood_tree.query(building_geom)

            max_overlap = 0
            for cand_idx in candidate_indices:
                flood_row = flood_gdf.iloc[cand_idx]
                flood_geom = flood_row.geometry

                if building_geom.intersects(flood_geom):
                    try:
                        intersection = building_geom.intersection(flood_geom)
                        intersection_area = intersection.area
                        building_area = building_geom.area
                        overlap_pct = (intersection_area / building_area * 100) if building_area > 0 else 0
                        max_overlap = max(max_overlap, overlap_pct)
                    except:
                        pass

            overlap_ratios.append(max_overlap)

        building_gdf['겹침비율'] = overlap_ratios
        
        # 홍수위험 등급 분류
        flood_risk = building_gdf['겹침비율'].apply(self.classify_flood_risk)
        building_gdf['홍수등급'] = flood_risk.apply(lambda x: x[0])
        building_gdf['홍수점수'] = flood_risk.apply(lambda x: x[1])

        # 통계 출력
        flood_buildings = building_gdf[building_gdf['겹침비율'] > 0]
        print(f"    홍수위험지역 교차 건물: {len(flood_buildings)}개")

        return building_gdf

    def save_results(self, gdf: gpd.GeoDataFrame, filename: str = None,
                     region_name: str = None, district_name: str = None,
                     min_age: int = None) -> Path:
        """분석 결과 저장 (EPSG:5186 좌표계)"""
        if gdf is None or len(gdf) == 0:
            print("저장할 데이터가 없습니다.")
            return None

        # 파일명 생성
        if filename is None:
            parts = ['통합위험분석']
            if region_name:
                parts.append(region_name)
            if district_name:
                parts.append(district_name)
            if min_age:
                parts.append(f'{min_age}년이상')
            parts.append(datetime.now().strftime('%Y%m%d'))
            filename = '_'.join(parts) + '.shp'

        output_path = self.output_path / filename

        # EPSG:5186 좌표계로 변환
        print(f"\n좌표계 변환 중 (EPSG:5186)...")
        gdf_5186 = gdf.copy()

        try:
            if gdf_5186.crs and gdf_5186.crs.to_epsg() != 5186:
                gdf_5186 = gdf_5186.to_crs(epsg=5186)
                print(f"  원본 좌표계: {gdf.crs} → EPSG:5186 변환 완료")
            elif not gdf_5186.crs:
                gdf_5186 = gdf_5186.set_crs(epsg=5186)
                print(f"  좌표계 설정: EPSG:5186")
        except Exception as e:
            print(f"  좌표계 변환 실패: {e}")
            print(f"  원본 좌표계로 저장합니다.")
            gdf_5186 = gdf.copy()

        # 모든 좌표를 EPSG:5186 기준으로 재계산
        print("  좌표 재계산 (EPSG:5186)...")
        gdf_5186['중심점X'] = gdf_5186.geometry.centroid.x
        gdf_5186['중심점Y'] = gdf_5186.geometry.centroid.y
        gdf_5186['경도'] = gdf_5186.geometry.centroid.x
        gdf_5186['위도'] = gdf_5186.geometry.centroid.y

        # SHP 저장 (cp949 인코딩)
        print(f"\n결과 저장 중: {output_path}")
        self._save_shp_with_encoding(gdf_5186, output_path, encoding='cp949')

        # CSV 요약 저장 (모든 분석 결과 포함)
        csv_path = output_path.with_suffix('.csv')
        summary_cols = [
            # 위치 정보
            '지역', '구군', '지역코드', '주소',
            '중심점X', '중심점Y', '경도', '위도',
            # 건물 정보
            '건축년도', '건물연령', '연령대',
            # 개별 위험도
            '노후등급', '노후점수',
            '겹침비율', '홍수등급', '홍수점수',
            '산사태거리', '산사태등급', '산사태점수',
            # 조합별 위험도
            '노후홍수점수', '노후홍수등급', '노후홍수코드',
            '홍수산사태점수', '홍수산사태등급', '홍수산사태코드',
            '노후산사태점수', '노후산사태등급', '노후산사태코드',
            # 종합 위험도
            '종합점수', '종합등급', '위험코드'
        ]

        # 존재하는 컬럼만 선택
        available_cols = [col for col in summary_cols if col in gdf_5186.columns]
        gdf_5186[available_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"CSV 요약 저장: {csv_path}")
        print(f"  - 포함 컬럼: {len(available_cols)}개")

        return output_path

    def print_summary(self, gdf: gpd.GeoDataFrame):
        """분석 결과 요약 출력"""
        if gdf is None or len(gdf) == 0:
            print("분석 결과가 없습니다.")
            return

        print("\n" + "=" * 70)
        print("              통합 위험분석 결과 요약")
        print("=" * 70)

        print(f"\n총 분석 건물 수: {len(gdf):,}개")
        print(f"분석 기준 연도: {self.current_year}년")

        # =========================================================================
        # 1. 개별 위험요소 분석 결과
        # =========================================================================
        print("\n" + "-" * 70)
        print("1. 개별 위험요소 분석")
        print("-" * 70)

        # 1-1. 노후위험도
        print("\n[1-1. 노후위험도]")
        print(f"  연령대별 분포:")
        age_counts = gdf['연령대'].value_counts()
        for age_cat, count in age_counts.items():
            pct = 100 * count / len(gdf)
            print(f"    {age_cat}: {count:,}개 ({pct:.1f}%)")

        print(f"\n  노후등급별 분포:")
        age_risk_counts = gdf['노후등급'].value_counts()
        for level, count in age_risk_counts.items():
            pct = 100 * count / len(gdf)
            print(f"    {level}: {count:,}개 ({pct:.1f}%)")

        avg_age = gdf['건물연령'].mean()
        print(f"\n  평균 건물연령: {avg_age:.1f}년")

        # 1-2. 홍수위험도
        if '홍수등급' in gdf.columns:
            print("\n[1-2. 홍수위험도]")
            flood_risk_counts = gdf['홍수등급'].value_counts()
            for level, count in flood_risk_counts.items():
                pct = 100 * count / len(gdf)
                print(f"    {level}: {count:,}개 ({pct:.1f}%)")

            flood_buildings = gdf[gdf['겹침비율'] > 0]
            print(f"\n  홍수위험지역 교차 건물: {len(flood_buildings):,}개")
            if len(flood_buildings) > 0:
                print(f"  평균 겹침비율: {flood_buildings['겹침비율'].mean():.1f}%")

        # 1-3. 산사태근접위험도
        if '산사태등급' in gdf.columns:
            print("\n[1-3. 산사태근접위험도]")
            landslide_counts = gdf['산사태등급'].value_counts()
            for level, count in landslide_counts.items():
                pct = 100 * count / len(gdf)
                print(f"    {level}: {count:,}개 ({pct:.1f}%)")

            valid_distances = gdf[gdf['산사태거리'] < 99999]['산사태거리']
            if len(valid_distances) > 0:
                print(f"\n  평균 산사태위험지역 거리: {valid_distances.mean():.1f}m")
                print(f"  최소 거리: {valid_distances.min():.1f}m")

        # =========================================================================
        # 2. 조합별 위험도 분석 결과
        # =========================================================================
        print("\n" + "-" * 70)
        print("2. 조합별 위험도 분석")
        print("-" * 70)

        # 2-1. 노후+홍수
        if '노후홍수등급' in gdf.columns:
            print("\n[2-1. 노후+홍수 복합위험]")
            nh_counts = gdf['노후홍수등급'].value_counts()
            for level, count in nh_counts.items():
                code = gdf[gdf['노후홍수등급'] == level]['노후홍수코드'].iloc[0]
                pct = 100 * count / len(gdf)
                print(f"    {code}등급 ({level}): {count:,}개 ({pct:.1f}%)")

        # 2-2. 홍수+산사태
        if '홍수산사태등급' in gdf.columns:
            print("\n[2-2. 홍수+산사태 복합위험]")
            hs_counts = gdf['홍수산사태등급'].value_counts()
            for level, count in hs_counts.items():
                code = gdf[gdf['홍수산사태등급'] == level]['홍수산사태코드'].iloc[0]
                pct = 100 * count / len(gdf)
                print(f"    {code}등급 ({level}): {count:,}개 ({pct:.1f}%)")

        # 2-3. 노후+산사태
        if '노후산사태등급' in gdf.columns:
            print("\n[2-3. 노후+산사태 복합위험]")
            ns_counts = gdf['노후산사태등급'].value_counts()
            for level, count in ns_counts.items():
                code = gdf[gdf['노후산사태등급'] == level]['노후산사태코드'].iloc[0]
                pct = 100 * count / len(gdf)
                print(f"    {code}등급 ({level}): {count:,}개 ({pct:.1f}%)")

        # =========================================================================
        # 3. 종합 위험도 분석 결과
        # =========================================================================
        print("\n" + "-" * 70)
        print("3. 종합 위험도 분석 (노후+홍수+산사태)")
        print("-" * 70)

        print("\n[종합위험등급별 건물 수]")
        combined_counts = gdf['종합등급'].value_counts()
        for level, count in combined_counts.items():
            code = gdf[gdf['종합등급'] == level]['위험코드'].iloc[0]
            pct = 100 * count / len(gdf)
            print(f"    {code}등급 ({level}): {count:,}개 ({pct:.1f}%)")

        # 최고 위험 건물
        max_score = gdf['종합점수'].max()
        max_risk = gdf[gdf['종합점수'] == max_score]
        print(f"\n  최고 위험 건물 (종합점수 {max_score}점): {len(max_risk):,}개")

        # 위험등급 D/E 건물 수 (각각 표시)
        d_grade = gdf[gdf['위험코드'] == 'D']
        e_grade = gdf[gdf['위험코드'] == 'E']
        d_count = len(d_grade)
        e_count = len(e_grade)
        total_de = d_count + e_count
        print(f"\n  [고위험 건물 상세]")
        print(f"    D등급 (경고): {d_count:,}개 ({100*d_count/len(gdf):.1f}%)")
        print(f"    E등급 (위험): {e_count:,}개 ({100*e_count/len(gdf):.1f}%)")
        print(f"    ─────────────────────────────")
        print(f"    D+E등급 합계: {total_de:,}개 ({100*total_de/len(gdf):.1f}%)")

        # =========================================================================
        # 4. 좌표 정보 현황
        # =========================================================================
        print("\n" + "-" * 70)
        print("4. 좌표 정보 현황 (EPSG:5186)")
        print("-" * 70)

        if '중심점X' in gdf.columns and '중심점Y' in gdf.columns:
            valid_coords = gdf[(gdf['중심점X'].notna()) & (gdf['중심점Y'].notna())]
            print(f"\n  좌표 추출 건물: {len(valid_coords):,}개")
            if len(valid_coords) > 0:
                print(f"  X 좌표 범위: {valid_coords['중심점X'].min():.2f} ~ {valid_coords['중심점X'].max():.2f}")
                print(f"  Y 좌표 범위: {valid_coords['중심점Y'].min():.2f} ~ {valid_coords['중심점Y'].max():.2f}")

        print("\n" + "=" * 70)


def find_similar_names(input_name: str, valid_names: list, threshold: float = 0.5) -> list:
    """유사한 이름 찾기 (오타 감지용)"""
    from difflib import SequenceMatcher

    similar = []
    for name in valid_names:
        ratio = SequenceMatcher(None, input_name, name).ratio()
        if ratio >= threshold:
            similar.append((name, ratio))

    # 유사도 순으로 정렬
    similar.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in similar[:3]]  # 상위 3개


def validate_region_input(analyzer, region_name: str) -> str:
    """광역지자체명 유효성 검사 및 오타 수정"""
    valid_regions = list(analyzer.region_mapping.keys())

    while region_name not in valid_regions:
        print(f"\n⚠️  '{region_name}'은(는) 올바르지 않은 지역명입니다.")

        # 유사한 지역명 제안
        similar = find_similar_names(region_name, valid_regions)
        if similar:
            print(f"   혹시 다음 중 하나를 의미하셨나요?")
            for i, name in enumerate(similar, 1):
                print(f"     {i}. {name}")

        print(f"\n   사용 가능한 지역: {', '.join(valid_regions)}")
        region_name = input("\n다시 입력하세요: ").strip()

        if not region_name:
            return None

    return region_name


def validate_district_input(analyzer, region_name: str, district_input: str) -> list:
    """구/군명 유효성 검사 및 오타 수정 (복수 입력 지원)"""
    valid_districts = list(analyzer.region_mapping[region_name]['districts'].keys())

    # 쉼표로 분리하여 복수 입력 처리
    input_districts = [d.strip() for d in district_input.split(',') if d.strip()]

    validated = []
    for district in input_districts:
        current_district = district

        while current_district not in valid_districts:
            print(f"\n⚠️  '{current_district}'은(는) '{region_name}'에 존재하지 않는 구/군명입니다.")

            # 유사한 구/군명 제안
            similar = find_similar_names(current_district, valid_districts)
            if similar:
                print(f"   혹시 다음 중 하나를 의미하셨나요?")
                for i, name in enumerate(similar, 1):
                    print(f"     {i}. {name}")

            print(f"\n   '{region_name}'의 구/군 목록:")
            for i, name in enumerate(valid_districts, 1):
                print(f"     {i}. {name}")

            choice = input(f"\n'{district}' 대신 입력할 구/군명 (건너뛰려면 Enter): ").strip()

            if not choice:
                print(f"   '{district}' 건너뜀")
                current_district = None
                break
            else:
                current_district = choice

        if current_district:
            validated.append(current_district)

    return validated


def interactive_mode():
    """대화형 모드로 실행 (복수 지역, 오타 감지, 광역 일괄 분석 지원)"""
    print("=" * 70)
    print("    건물 노후위험도 × 홍수위험 × 산사태근접위험 통합 분석")
    print("=" * 70)
    print(f"분석 기준 연도: {datetime.now().year}년")
    print("좌표계: EPSG:5186 (Korea 2000 / Central Belt)")

    # 기본 경로 설정
    base_path = r"C:\Users\user\Downloads\kescoaitest"

    if not os.path.exists(base_path):
        print(f"기본 경로를 찾을 수 없습니다: {base_path}")
        base_path = input("데이터 폴더 경로를 입력하세요: ").strip()

    analyzer = BuildingMultiRiskAnalyzer(base_path)

    # 지역 목록 표시
    analyzer.list_available_regions()

    # =========================================================================
    # 1. 광역지자체 선택 (복수 입력 지원)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[1단계] 광역지자체 선택")
    print("-" * 70)
    print("  - 복수 선택 가능: 쉼표(,)로 구분 (예: 서울, 부산)")
    print("  - 단일 선택 후 구/군 지정 가능")

    region_input = input("\n분석할 광역지자체를 입력하세요: ").strip()

    if not region_input:
        print("지역이 입력되지 않았습니다.")
        return

    # 복수 광역지자체 처리
    region_names = [r.strip() for r in region_input.split(',') if r.strip()]
    validated_regions = []

    for region in region_names:
        validated = validate_region_input(analyzer, region)
        if validated:
            validated_regions.append(validated)

    if not validated_regions:
        print("유효한 지역이 없습니다.")
        return

    print(f"\n✓ 선택된 광역지자체: {', '.join(validated_regions)}")

    # =========================================================================
    # 2. 구/군 선택 (복수 입력 지원)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2단계] 구/군 선택")
    print("-" * 70)
    print("  - 전체 분석: Enter 입력")
    print("  - 복수 선택: 쉼표(,)로 구분 (예: 전주시 완산구, 전주시 덕진구)")

    districts_by_region = {}

    for region in validated_regions:
        print(f"\n[{region}] 구/군 목록:")
        district_list = list(analyzer.region_mapping[region]['districts'].keys())
        for i, d in enumerate(district_list, 1):
            print(f"  {i}. {d}")

        if len(validated_regions) == 1:
            district_input = input(f"\n'{region}'에서 분석할 구/군 (전체는 Enter): ").strip()
        else:
            district_input = input(f"\n'{region}'에서 분석할 구/군 (전체는 Enter, 건너뛰기는 'skip'): ").strip()

        if district_input.lower() == 'skip':
            print(f"  '{region}' 건너뜀")
            continue
        elif not district_input:
            # 전체 분석
            districts_by_region[region] = None
            print(f"  ✓ '{region}' 전체 분석")
        else:
            # 개별 구/군 선택
            validated_districts = validate_district_input(analyzer, region, district_input)
            if validated_districts:
                districts_by_region[region] = validated_districts
                print(f"  ✓ 선택된 구/군: {', '.join(validated_districts)}")
            else:
                print(f"  '{region}'에서 유효한 구/군이 없어 전체 분석으로 진행합니다.")
                districts_by_region[region] = None

    if not districts_by_region:
        print("분석할 지역이 없습니다.")
        return

    # =========================================================================
    # 3. 연령 필터 설정
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3단계] 연령 필터 설정")
    print("-" * 70)

    min_age_input = input("최소 건물연령 (예: 30, 미설정은 Enter): ").strip()
    min_age = int(min_age_input) if min_age_input and min_age_input.isdigit() else None

    max_age_input = input("최대 건물연령 (예: 50, 미설정은 Enter): ").strip()
    max_age = int(max_age_input) if max_age_input and max_age_input.isdigit() else None

    if min_age:
        print(f"  ✓ 최소 연령: {min_age}년 이상")
    if max_age:
        print(f"  ✓ 최대 연령: {max_age}년 이하")

    # =========================================================================
    # 4. 분석 옵션
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4단계] 분석 옵션")
    print("-" * 70)

    # 홍수위험 분석 포함 여부
    while True:
        flood_input = input("홍수위험 분석 포함? (y/n, 기본 y): ").strip().lower()
        if flood_input in ['y', 'n', '']:
            include_flood = flood_input != 'n'
            break
        else:
            print("  ⚠️ 'y' 또는 'n'만 입력해주세요.")

    # 산사태근접위험 분석 포함 여부
    while True:
        landslide_input = input("산사태근접위험 분석 포함? (y/n, 기본 y): ").strip().lower()
        if landslide_input in ['y', 'n', '']:
            include_landslide = landslide_input != 'n'
            break
        else:
            print("  ⚠️ 'y' 또는 'n'만 입력해주세요.")

    print(f"  ✓ 홍수위험 분석: {'포함' if include_flood else '제외'}")
    print(f"  ✓ 산사태근접위험 분석: {'포함' if include_landslide else '제외'}")

    # =========================================================================
    # 5. 분석 실행
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5단계] 분석 실행")
    print("=" * 70)

    all_results = []
    result_info = []  # (result, region, district) 튜플 저장

    for region, districts in districts_by_region.items():
        if districts is None:
            # 전체 분석
            print(f"\n▶ {region} 전체 분석 시작...")
            result = analyzer.analyze_region(
                region, None, min_age, max_age,
                include_flood, include_landslide
            )
            if result is not None and len(result) > 0:
                all_results.append(result)
                result_info.append((result, region, None))
        else:
            # 개별 구/군 분석
            for district in districts:
                print(f"\n▶ {region} {district} 분석 시작...")
                result = analyzer.analyze_region(
                    region, district, min_age, max_age,
                    include_flood, include_landslide
                )
                if result is not None and len(result) > 0:
                    all_results.append(result)
                    result_info.append((result, region, district))

    # =========================================================================
    # 6. 결과 저장
    # =========================================================================
    if not all_results:
        print("\n분석 결과가 없습니다.")
        # 추가 작업 여부 확인
        print("\n" + "=" * 70)
        while True:
            continue_input = input("추가 분석을 진행하시겠습니까? (y/n): ").strip().lower()
            if continue_input == 'y':
                print("\n" + "=" * 70)
                print("새로운 분석을 시작합니다...")
                interactive_mode()
                return
            elif continue_input == 'n':
                print("\n프로그램을 종료합니다. 감사합니다!")
                return
            else:
                print("  ⚠️ 'y' 또는 'n'만 입력해주세요.")

    print("\n" + "=" * 70)
    print("[6단계] 결과 저장")
    print("=" * 70)

    total_buildings = sum(len(r) for r in all_results)
    print(f"\n분석 완료: {len(all_results)}개 지역, {total_buildings:,}개 건물")

    # 2개 이상의 결과가 있으면 저장 방식 선택
    if len(all_results) >= 2:
        print("\n저장 방식을 선택해주세요:")
        print("  1. 하나의 파일로 합쳐서 저장")
        print("  2. 각 지역별로 따로 저장")

        while True:
            save_choice = input("\n선택 (1 또는 2): ").strip()
            if save_choice == '1':
                # 하나로 합쳐서 저장
                print("\n모든 결과를 하나의 파일로 병합 중...")
                combined_result = pd.concat(all_results, ignore_index=True)
                combined_gdf = gpd.GeoDataFrame(combined_result, crs=all_results[0].crs)

                # 파일명 생성 (지역명들 조합)
                region_names = list(set(r for _, r, _ in result_info))
                if len(region_names) <= 3:
                    region_str = '_'.join(region_names)
                else:
                    region_str = f"{region_names[0]}외{len(region_names)-1}개지역"

                filename = f"통합위험분석_{region_str}"
                if min_age:
                    filename += f"_{min_age}년이상"
                filename += f"_{datetime.now().strftime('%Y%m%d')}.shp"

                output_path = analyzer.save_results(combined_gdf, filename=filename)
                print(f"\n✓ 통합 저장 완료: {output_path}")

                # 요약 출력
                analyzer.print_summary(combined_gdf)
                break

            elif save_choice == '2':
                # 각각 저장
                print("\n각 지역별로 저장 중...")
                for result, region, district in result_info:
                    output_path = analyzer.save_results(
                        result, region_name=region, district_name=district, min_age=min_age
                    )
                    district_str = district if district else "전체"
                    print(f"  ✓ {region} {district_str}: {output_path.name}")
                break

            else:
                print("  ⚠️ '1' 또는 '2'만 입력해주세요.")
    else:
        # 단일 결과는 바로 저장
        result, region, district = result_info[0]
        output_path = analyzer.save_results(
            result, region_name=region, district_name=district, min_age=min_age
        )
        print(f"\n✓ 저장 완료: {output_path}")

        # 요약 출력
        analyzer.print_summary(result)

    # =========================================================================
    # 7. 최종 요약
    # =========================================================================
    print("\n" + "=" * 70)
    print("분석 완료")
    print("=" * 70)
    print(f"총 분석 지역: {len(all_results)}개")
    print(f"총 분석 건물: {total_buildings:,}개")
    print(f"결과 저장 위치: {analyzer.output_path}")

    # =========================================================================
    # 8. 추가 작업 여부 확인
    # =========================================================================
    print("\n" + "=" * 70)
    while True:
        continue_input = input("추가 분석을 진행하시겠습니까? (y/n): ").strip().lower()
        if continue_input == 'y':
            print("\n" + "=" * 70)
            print("새로운 분석을 시작합니다...")
            interactive_mode()  # 재귀 호출로 다시 시작
            return
        elif continue_input == 'n':
            print("\n프로그램을 종료합니다. 감사합니다!")
            return
        else:
            print("  ⚠️ 'y' 또는 'n'만 입력해주세요.")


def quick_analyze(region_name: str, district_names=None,
                  min_age: int = None, max_age: int = None,
                  include_flood: bool = True, include_landslide: bool = True):
    """
    빠른 분석 실행 (스크립트에서 직접 호출용)

    Args:
        region_name: 지역명 (서울, 부산, 전북 등)
        district_names: 구/군명 (None이면 전체, str이면 단일, list면 복수)
        min_age: 최소 건물연령 필터
        max_age: 최대 건물연령 필터
        include_flood: 홍수위험 분석 포함 여부
        include_landslide: 산사태근접위험 분석 포함 여부

    Returns:
        분석 결과 GeoDataFrame (지도 시각화용 좌표 포함)

    출력 파일:
        - SHP: EPSG:5186 좌표계
        - CSV: 모든 분석 결과 포함

    출력 컬럼:
        - 위치: 지역, 구군, 지역코드, 주소, 중심점X, 중심점Y, 경도, 위도 (모두 EPSG:5186)
        - 개별위험: 노후등급/점수, 홍수등급/점수, 산사태등급/점수/거리
        - 조합위험: 노후홍수, 홍수산사태, 노후산사태 등급/점수
        - 종합위험: 종합점수, 종합등급, 위험코드

    Examples:
        # 단일 구/군 분석
        quick_analyze("전북", "전주시 완산구", min_age=30)

        # 복수 구/군 분석
        quick_analyze("전북", ["전주시 완산구", "전주시 덕진구"], min_age=30)

        # 광역지자체 전체 분석
        quick_analyze("전북", min_age=30)
    """
    base_path = r"C:\Users\user\Downloads\kescoaitest"
    analyzer = BuildingMultiRiskAnalyzer(base_path)

    all_results = []

    # district_names 처리: None, str, list 모두 지원
    if district_names is None:
        # 전체 분석
        districts = [None]
    elif isinstance(district_names, str):
        # 단일 구/군
        districts = [district_names]
    else:
        # 리스트
        districts = district_names

    for district in districts:
        result = analyzer.analyze_region(
            region_name, district, min_age, max_age,
            include_flood, include_landslide
        )

        if result is not None and len(result) > 0:
            analyzer.print_summary(result)
            analyzer.save_results(result, region_name=region_name,
                                district_name=district, min_age=min_age)
            all_results.append(result)

    # 복수 결과 병합
    if len(all_results) > 1:
        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)
        return gpd.GeoDataFrame(combined, crs=all_results[0].crs)
    elif len(all_results) == 1:
        return all_results[0]
    else:
        return None


def batch_analyze(regions: dict, min_age: int = None, max_age: int = None,
                  include_flood: bool = True, include_landslide: bool = True):
    """
    복수 광역지자체 일괄 분석

    Args:
        regions: {지역명: 구/군 리스트 또는 None} 딕셔너리
        min_age: 최소 건물연령 필터
        max_age: 최대 건물연령 필터
        include_flood: 홍수위험 분석 포함 여부
        include_landslide: 산사태근접위험 분석 포함 여부

    Examples:
        # 복수 광역지자체, 복수 구/군
        batch_analyze({
            "전북": ["전주시 완산구", "전주시 덕진구"],
            "서울": ["종로구", "중구"]
        }, min_age=30)

        # 광역지자체 전체 분석
        batch_analyze({
            "전북": None,  # 전북 전체
            "부산": None   # 부산 전체
        }, min_age=30)
    """
    all_results = []

    for region_name, district_names in regions.items():
        result = quick_analyze(
            region_name, district_names, min_age, max_age,
            include_flood, include_landslide
        )
        if result is not None:
            all_results.append(result)

    print(f"\n{'='*70}")
    print(f"일괄 분석 완료: {len(regions)}개 광역지자체")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    # 대화형 모드로 실행
    interactive_mode()

    # 또는 직접 호출 예시:

    # 1. 단일 구/군 분석
    # quick_analyze("전북", "전주시 완산구", min_age=30)

    # 2. 복수 구/군 분석
    # quick_analyze("전북", ["전주시 완산구", "전주시 덕진구"], min_age=30)

    # 3. 광역지자체 전체 분석
    # quick_analyze("전북", min_age=30)

    # 4. 복수 광역지자체 일괄 분석
    # batch_analyze({
    #     "전북": ["전주시 완산구", "전주시 덕진구"],
    #     "서울": None  # 서울 전체
    # }, min_age=30)
