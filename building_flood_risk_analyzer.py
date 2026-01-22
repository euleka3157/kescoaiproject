"""
건물 노후위험도 × 홍수위험 통합 분석 프로그램
==============================================
목적: 건물연령(A23 착공일자 기반) + 홍수위험을 결합하여
      종합 위험등급을 산출하고 필터링된 결과를 SHP로 출력

기능:
- A23(착공일자)에서 당해연도를 빼서 건물연령 계산
- 노후위험도 등급: 0-5, 5-10, 10-20, 20-30, 30-50, 50-100년
- 홍수위험 + 노후위험 결합 종합위험등급 산출
- 사용자 입력: 지역 선택, 연령 필터링
- cp949 인코딩 지원 (한국 SHP 파일)

사용법:
    python building_flood_risk_analyzer.py

필요 라이브러리:
    pip install geopandas shapely fiona pyproj rtree pandas numpy
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
    import numpy as np
except ImportError as e:
    print(f"필요한 라이브러리가 없습니다: {e}")
    print("다음 명령어로 설치해주세요:")
    print("pip install geopandas shapely fiona pyproj rtree pandas numpy")
    sys.exit(1)


class BuildingFloodRiskAnalyzer:
    """건물 노후위험도 × 홍수위험 통합 분석 클래스"""

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
        (0, 10, '낮음', 1),
        (10, 30, '보통', 2),
        (30, 50, '높음', 3),
        (50, 100, '매우높음', 4)
    ]

    # 종합위험등급 기준 (점수 합계)
    COMBINED_RISK_LEVELS = [
        (0, 3, '안전', 'A'),
        (3, 5, '관심', 'B'),
        (5, 7, '주의', 'C'),
        (7, 9, '경고', 'D'),
        (9, 12, '위험', 'E')
    ]

    def __init__(self, base_path: str):
        """
        Args:
            base_path: 데이터 기본 경로
        """
        self.base_path = Path(base_path)
        self.building_path = self.base_path / "건물연령"
        self.flood_path = self.base_path / "홍수위험"
        self.output_path = self.base_path / "분석결과"
        self.current_year = datetime.now().year

        # 출력 폴더 생성
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 지역 매핑 정보
        self.region_mapping = self._build_region_mapping()

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
        # cpg 파일로 인코딩 지정
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

    def calculate_building_age(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """A23(착공일자)에서 건물연령 계산"""
        gdf = gdf.copy()

        def extract_year(date_val):
            """날짜에서 연도 추출"""
            if pd.isna(date_val):
                return None
            try:
                if isinstance(date_val, str):
                    # 'YYYY-MM-DD' 또는 'YYYYMMDD' 형식
                    year_str = date_val[:4]
                    return int(year_str)
                elif hasattr(date_val, 'year'):
                    return date_val.year
                else:
                    return int(str(date_val)[:4])
            except:
                return None

        # 착공연도 추출
        gdf['건축년도'] = gdf['A23'].apply(extract_year)

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
        """종합위험등급 분류"""
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

    def analyze_region(self, region_name: str, district_name: str = None,
                       min_age: int = None, max_age: int = None) -> gpd.GeoDataFrame:
        """
        특정 지역 분석 수행

        Args:
            region_name: 지역명 (서울, 부산, 전북 등)
            district_name: 구/군명 (None이면 해당 지역 전체)
            min_age: 최소 건물연령 필터 (이 연령 이상만 포함)
            max_age: 최대 건물연령 필터 (이 연령 이하만 포함)

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
                dist_info, region_name, dist_name, min_age, max_age
            )

            if result is not None and len(result) > 0:
                all_results.append(result)

        if all_results:
            combined_result = pd.concat(all_results, ignore_index=True)
            return gpd.GeoDataFrame(combined_result, crs=all_results[0].crs)

        return None

    def _analyze_single_district(self, dist_info: dict, region_name: str,
                                  district_name: str, min_age: int, max_age: int) -> gpd.GeoDataFrame:
        """단일 구/군 분석"""
        building_shp = dist_info['building_shp']
        region_code = dist_info['code']

        # 홍수위험 SHP 찾기
        flood_shp = self.find_flood_shp(region_code)

        if not flood_shp:
            print(f"  홍수위험 데이터 없음 (코드: {region_code})")
            return None

        print(f"  건물연령 파일: {building_shp.name}")
        print(f"  홍수위험 파일: {flood_shp.name}")

        # 데이터 로드
        print("  데이터 로딩 중...")
        building_gdf = self._read_shp_with_encoding(building_shp)
        flood_gdf = self._read_shp_with_encoding(flood_shp)

        print(f"  건물 수: {len(building_gdf)}, 홍수위험구역 수: {len(flood_gdf)}")

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

        # 홍수위험 교차 분석
        print("  홍수위험 교차 분석 중...")
        result_gdf = self._analyze_flood_intersection(building_gdf, flood_gdf)

        if result_gdf is None or len(result_gdf) == 0:
            print("  홍수위험 지역과 교차하는 건물 없음")
            return None

        # 노후위험도 등급 계산
        print("  위험등급 계산 중...")
        result_gdf = self._calculate_risk_levels(result_gdf)

        # 지역 정보 추가
        result_gdf['지역'] = region_name
        result_gdf['구군'] = district_name
        result_gdf['지역코드'] = region_code

        print(f"  분석 완료: {len(result_gdf)}개 위험 건물 발견")

        return result_gdf

    def _analyze_flood_intersection(self, building_gdf: gpd.GeoDataFrame,
                                     flood_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """홍수위험 교차 분석"""
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

        # 교차 분석
        intersecting_data = []
        total = len(building_gdf)

        for idx, (building_idx, building_row) in enumerate(building_gdf.iterrows()):
            if idx % 1000 == 0 and idx > 0:
                print(f"    진행률: {idx}/{total} ({100*idx/total:.1f}%)")

            building_geom = building_row.geometry
            candidate_indices = flood_tree.query(building_geom)

            for cand_idx in candidate_indices:
                flood_row = flood_gdf.iloc[cand_idx]
                flood_geom = flood_row.geometry

                if building_geom.intersects(flood_geom):
                    try:
                        intersection = building_geom.intersection(flood_geom)
                        intersection_area = intersection.area
                        building_area = building_geom.area
                        overlap_pct = (intersection_area / building_area * 100) if building_area > 0 else 0
                    except:
                        intersection_area = 0
                        overlap_pct = 0

                    row_data = building_row.to_dict()
                    row_data['교차면적'] = round(intersection_area, 2)
                    row_data['겹침비율'] = round(overlap_pct, 2)

                    # 홍수위험 정보 추가
                    for col in flood_gdf.columns:
                        if col != 'geometry':
                            row_data[f'홍수_{col}'] = flood_row[col]

                    intersecting_data.append(row_data)
                    break

        if intersecting_data:
            return gpd.GeoDataFrame(intersecting_data, crs=building_gdf.crs)
        return None

    def _calculate_risk_levels(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """위험등급 계산"""
        gdf = gdf.copy()

        # 연령대 카테고리
        gdf['연령대'] = gdf['건물연령'].apply(self.get_age_category)

        # 노후위험도 등급
        age_risk = gdf['건물연령'].apply(self.classify_age_risk)
        gdf['노후등급'] = age_risk.apply(lambda x: x[0])
        gdf['노후점수'] = age_risk.apply(lambda x: x[1])

        # 홍수위험 등급
        flood_risk = gdf['겹침비율'].apply(self.classify_flood_risk)
        gdf['홍수등급'] = flood_risk.apply(lambda x: x[0])
        gdf['홍수점수'] = flood_risk.apply(lambda x: x[1])

        # 종합위험점수 및 등급
        gdf['종합점수'] = gdf['노후점수'] + gdf['홍수점수']
        combined_risk = gdf['종합점수'].apply(self.classify_combined_risk)
        gdf['종합등급'] = combined_risk.apply(lambda x: x[0])
        gdf['위험코드'] = combined_risk.apply(lambda x: x[1])

        return gdf

    def save_results(self, gdf: gpd.GeoDataFrame, filename: str = None,
                     region_name: str = None, district_name: str = None,
                     min_age: int = None) -> Path:
        """분석 결과 저장"""
        if gdf is None or len(gdf) == 0:
            print("저장할 데이터가 없습니다.")
            return None

        # 파일명 생성
        if filename is None:
            parts = ['위험건물분석']
            if region_name:
                parts.append(region_name)
            if district_name:
                parts.append(district_name)
            if min_age:
                parts.append(f'{min_age}년이상')
            parts.append(datetime.now().strftime('%Y%m%d'))
            filename = '_'.join(parts) + '.shp'

        output_path = self.output_path / filename

        # SHP 저장 (cp949 인코딩)
        print(f"\n결과 저장 중: {output_path}")
        self._save_shp_with_encoding(gdf, output_path, encoding='cp949')

        # CSV 요약 저장
        csv_path = output_path.with_suffix('.csv')
        summary_cols = ['지역', '구군', '지역코드', '건축년도', '건물연령', '연령대',
                       '노후등급', '노후점수', '겹침비율', '홍수등급', '홍수점수',
                       '종합점수', '종합등급', '위험코드']

        # 존재하는 컬럼만 선택
        available_cols = [col for col in summary_cols if col in gdf.columns]
        gdf[available_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"CSV 요약 저장: {csv_path}")

        return output_path

    def print_summary(self, gdf: gpd.GeoDataFrame):
        """분석 결과 요약 출력"""
        if gdf is None or len(gdf) == 0:
            print("분석 결과가 없습니다.")
            return

        print("\n" + "=" * 60)
        print("분석 결과 요약")
        print("=" * 60)

        print(f"\n총 위험 건물 수: {len(gdf)}개")
        print(f"분석 기준 연도: {self.current_year}년")

        # 연령대별 통계
        print("\n[연령대별 건물 수]")
        age_counts = gdf['연령대'].value_counts()
        for age_cat, count in age_counts.items():
            print(f"  {age_cat}: {count}개")

        # 노후등급별 통계
        print("\n[노후위험등급별 건물 수]")
        age_risk_counts = gdf['노후등급'].value_counts()
        for level, count in age_risk_counts.items():
            print(f"  {level}: {count}개")

        # 홍수등급별 통계
        print("\n[홍수위험등급별 건물 수]")
        flood_risk_counts = gdf['홍수등급'].value_counts()
        for level, count in flood_risk_counts.items():
            print(f"  {level}: {count}개")

        # 종합등급별 통계
        print("\n[종합위험등급별 건물 수]")
        combined_counts = gdf['종합등급'].value_counts()
        for level, count in combined_counts.items():
            code = gdf[gdf['종합등급'] == level]['위험코드'].iloc[0]
            print(f"  {code}등급 ({level}): {count}개")

        # 평균 연령
        avg_age = gdf['건물연령'].mean()
        print(f"\n평균 건물연령: {avg_age:.1f}년")

        # 최고 위험 건물
        max_risk = gdf[gdf['종합점수'] == gdf['종합점수'].max()]
        print(f"\n최고 위험 건물 수 (종합점수 {max_risk['종합점수'].iloc[0]}): {len(max_risk)}개")


def interactive_mode():
    """대화형 모드로 실행"""
    print("=" * 60)
    print("건물 노후위험도 × 홍수위험 통합 분석 프로그램")
    print("=" * 60)
    print(f"분석 기준 연도: {datetime.now().year}년")

    # 기본 경로 설정
    base_path = r"C:\Users\user\Downloads\kescoaitest"

    if not os.path.exists(base_path):
        print(f"기본 경로를 찾을 수 없습니다: {base_path}")
        base_path = input("데이터 폴더 경로를 입력하세요: ").strip()

    analyzer = BuildingFloodRiskAnalyzer(base_path)

    # 지역 목록 표시
    analyzer.list_available_regions()

    # 지역 선택
    print("\n" + "-" * 40)
    region_name = input("분석할 지역을 입력하세요 (예: 서울, 부산, 전북): ").strip()

    if not region_name:
        print("지역이 입력되지 않았습니다.")
        return

    # 구/군 선택 (선택사항)
    district_name = input("분석할 구/군을 입력하세요 (전체 분석은 Enter): ").strip()
    if not district_name:
        district_name = None

    # 연령 필터 설정
    print("\n[연령 필터 설정]")
    min_age_input = input("최소 건물연령 (예: 30, 미설정은 Enter): ").strip()
    min_age = int(min_age_input) if min_age_input else None

    max_age_input = input("최대 건물연령 (예: 50, 미설정은 Enter): ").strip()
    max_age = int(max_age_input) if max_age_input else None

    # 분석 실행
    print("\n분석을 시작합니다...")
    result = analyzer.analyze_region(region_name, district_name, min_age, max_age)

    if result is not None and len(result) > 0:
        # 결과 요약 출력
        analyzer.print_summary(result)

        # 결과 저장
        output_path = analyzer.save_results(
            result,
            region_name=region_name,
            district_name=district_name,
            min_age=min_age
        )

        print(f"\n분석 완료! 결과 파일: {output_path}")
    else:
        print("\n분석 결과가 없습니다.")


def quick_analyze(region_name: str, district_name: str = None,
                  min_age: int = None, max_age: int = None):
    """빠른 분석 실행 (스크립트에서 직접 호출용)"""
    base_path = r"C:\Users\user\Downloads\kescoaitest"
    analyzer = BuildingFloodRiskAnalyzer(base_path)

    result = analyzer.analyze_region(region_name, district_name, min_age, max_age)

    if result is not None and len(result) > 0:
        analyzer.print_summary(result)
        analyzer.save_results(result, region_name=region_name,
                            district_name=district_name, min_age=min_age)

    return result


if __name__ == "__main__":
    # 대화형 모드로 실행
    interactive_mode()

    # 또는 직접 호출 예시:
    # quick_analyze("전북", "전주시 완산구", min_age=30)
    # quick_analyze("서울", min_age=20)
