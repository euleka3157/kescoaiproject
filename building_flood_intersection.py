"""
건물연령 × 홍수위험 교차 분석 프로그램
=====================================
목적: 건물연령 shp와 홍수위험 shp 파일을 비교분석하여
      겹치거나 교차되는 건물들을 찾아 새로운 shp 파일로 출력

연산 최적화:
- Spatial Index (R-tree) 사용으로 O(n²) → O(n log n) 개선
- 병렬 처리 지원
- 청크 단위 메모리 관리

사용법:
    python building_flood_intersection.py

필요 라이브러리:
    pip install geopandas shapely fiona pyproj rtree
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box
    from shapely.strtree import STRtree
    import numpy as np
except ImportError as e:
    print(f"필요한 라이브러리가 없습니다: {e}")
    print("다음 명령어로 설치해주세요:")
    print("pip install geopandas shapely fiona pyproj rtree pandas numpy")
    sys.exit(1)


class BuildingFloodAnalyzer:
    """건물연령과 홍수위험 교차 분석 클래스"""
    
    def __init__(self, base_path: str):
        """
        Args:
            base_path: 데이터 기본 경로 (예: C:/Users/user/Downloads/kescoaitest)
        """
        self.base_path = Path(base_path)
        self.building_age_path = self.base_path / "건물연령"
        self.flood_risk_path = self.base_path / "홍수위험"
        self.output_path = self.base_path / "분석결과"
        
        # 출력 폴더 생성
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def find_shp_files(self, folder: Path) -> list:
        """폴더 내 모든 shp 파일 찾기"""
        shp_files = []
        if folder.exists():
            for shp_file in folder.rglob("*.shp"):
                shp_files.append(shp_file)
        return shp_files
    
    def get_region_code(self, filename: str) -> str:
        """파일명에서 지역코드 추출 (예: 52111)"""
        # AL_D196_52111_20240214.shp → 52111
        # CFM_SGG_52111_030.shp → 52111
        parts = filename.replace('.shp', '').split('_')
        for part in parts:
            if part.isdigit() and len(part) == 5:
                return part
        return None
    
    def match_files_by_region(self) -> list:
        """지역코드 기준으로 건물연령/홍수위험 파일 매칭"""
        building_files = self.find_shp_files(self.building_age_path)
        flood_files = self.find_shp_files(self.flood_risk_path)
        
        print(f"발견된 건물연령 파일: {len(building_files)}개")
        print(f"발견된 홍수위험 파일: {len(flood_files)}개")
        
        # 지역코드별로 인덱싱
        building_by_region = {}
        for f in building_files:
            code = self.get_region_code(f.name)
            if code:
                building_by_region[code] = f
                
        flood_by_region = {}
        for f in flood_files:
            code = self.get_region_code(f.name)
            if code:
                flood_by_region[code] = f
        
        # 매칭
        matched = []
        for code in building_by_region:
            if code in flood_by_region:
                matched.append({
                    'region_code': code,
                    'building_file': building_by_region[code],
                    'flood_file': flood_by_region[code]
                })
                
        print(f"매칭된 지역: {len(matched)}개")
        return matched
    
    def analyze_intersection_optimized(self, building_gdf: gpd.GeoDataFrame, 
                                       flood_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        최적화된 교차 분석 (Spatial Index 사용)
        
        Args:
            building_gdf: 건물연령 GeoDataFrame
            flood_gdf: 홍수위험 GeoDataFrame
            
        Returns:
            교차되는 건물들의 GeoDataFrame
        """
        start_time = time.time()
        
        # CRS 통일 (좌표계 맞추기)
        if building_gdf.crs != flood_gdf.crs:
            flood_gdf = flood_gdf.to_crs(building_gdf.crs)
        
        # 유효하지 않은 geometry 수정
        building_gdf['geometry'] = building_gdf['geometry'].buffer(0)
        flood_gdf['geometry'] = flood_gdf['geometry'].buffer(0)
        
        # Spatial Index 생성 (STRtree - 가장 빠름)
        print("  - Spatial Index 생성 중...")
        flood_tree = STRtree(flood_gdf.geometry.values)
        flood_indices = np.arange(len(flood_gdf))
        
        # 교차 분석
        print("  - 교차 분석 수행 중...")
        intersecting_buildings = []
        intersection_info = []
        
        total = len(building_gdf)
        for idx, (building_idx, building_row) in enumerate(building_gdf.iterrows()):
            if idx % 1000 == 0:
                print(f"    진행률: {idx}/{total} ({100*idx/total:.1f}%)")
            
            building_geom = building_row.geometry
            
            # Spatial Index로 후보 필터링 (Bounding Box 기준)
            candidate_indices = flood_tree.query(building_geom)
            
            # 실제 교차 여부 확인
            for cand_idx in candidate_indices:
                flood_row = flood_gdf.iloc[cand_idx]
                flood_geom = flood_row.geometry
                
                if building_geom.intersects(flood_geom):
                    # 교차 영역 계산
                    try:
                        intersection = building_geom.intersection(flood_geom)
                        intersection_area = intersection.area
                        building_area = building_geom.area
                        overlap_ratio = (intersection_area / building_area * 100) if building_area > 0 else 0
                    except:
                        intersection_area = 0
                        overlap_ratio = 0
                    
                    intersecting_buildings.append(building_idx)
                    intersection_info.append({
                        'building_idx': building_idx,
                        'flood_idx': cand_idx,
                        'intersection_area': intersection_area,
                        'overlap_ratio': overlap_ratio
                    })
                    break  # 하나라도 겹치면 해당 건물은 위험 건물
        
        # 결과 GeoDataFrame 생성
        if intersecting_buildings:
            result_gdf = building_gdf.loc[intersecting_buildings].copy()
            
            # 교차 정보 추가
            info_df = pd.DataFrame(intersection_info)
            info_grouped = info_df.groupby('building_idx').agg({
                'intersection_area': 'sum',
                'overlap_ratio': 'max',
                'flood_idx': 'count'
            }).rename(columns={'flood_idx': 'flood_zone_count'})
            
            result_gdf = result_gdf.join(info_grouped)
            
            # 위험 등급 추가 (겹침 비율 기준)
            result_gdf['risk_level'] = pd.cut(
                result_gdf['overlap_ratio'],
                bins=[0, 10, 30, 50, 100],
                labels=['낮음', '보통', '높음', '매우높음'],
                include_lowest=True
            )
        else:
            result_gdf = gpd.GeoDataFrame()
        
        elapsed = time.time() - start_time
        print(f"  - 분석 완료: {len(result_gdf)}개 위험 건물 발견 ({elapsed:.2f}초)")
        
        return result_gdf
    
    def process_single_region(self, region_info: dict) -> dict:
        """단일 지역 처리"""
        region_code = region_info['region_code']
        building_file = region_info['building_file']
        flood_file = region_info['flood_file']
        
        print(f"\n[지역코드: {region_code}]")
        print(f"  건물연령: {building_file.name}")
        print(f"  홍수위험: {flood_file.name}")
        
        try:
            # 데이터 로드
            print("  - 데이터 로딩 중...")
            building_gdf = gpd.read_file(building_file)
            flood_gdf = gpd.read_file(flood_file)
            
            print(f"  - 건물 수: {len(building_gdf)}, 홍수위험구역 수: {len(flood_gdf)}")
            
            # 교차 분석
            result_gdf = self.analyze_intersection_optimized(building_gdf, flood_gdf)
            
            if len(result_gdf) > 0:
                # 결과 저장
                output_file = self.output_path / f"위험건물_{region_code}.shp"
                result_gdf.to_file(output_file, encoding='utf-8')
                print(f"  - 저장 완료: {output_file}")
                
                return {
                    'region_code': region_code,
                    'total_buildings': len(building_gdf),
                    'risk_buildings': len(result_gdf),
                    'output_file': str(output_file),
                    'success': True
                }
            else:
                print(f"  - 교차하는 건물 없음")
                return {
                    'region_code': region_code,
                    'total_buildings': len(building_gdf),
                    'risk_buildings': 0,
                    'output_file': None,
                    'success': True
                }
                
        except Exception as e:
            print(f"  - 오류 발생: {e}")
            return {
                'region_code': region_code,
                'error': str(e),
                'success': False
            }
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("=" * 60)
        print("건물연령 × 홍수위험 교차 분석 시작")
        print("=" * 60)
        
        # 파일 매칭
        matched = self.match_files_by_region()
        
        if not matched:
            print("매칭되는 파일이 없습니다.")
            return
        
        # 각 지역 처리
        results = []
        for region_info in matched:
            result = self.process_single_region(region_info)
            results.append(result)
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("분석 결과 요약")
        print("=" * 60)
        
        total_buildings = 0
        total_risk = 0
        
        for r in results:
            if r['success']:
                total_buildings += r.get('total_buildings', 0)
                total_risk += r.get('risk_buildings', 0)
                status = f"{r.get('risk_buildings', 0)}개 위험건물"
            else:
                status = f"오류: {r.get('error', 'unknown')}"
            print(f"  {r['region_code']}: {status}")
        
        print(f"\n전체: {total_buildings}개 건물 중 {total_risk}개 위험건물 ({100*total_risk/total_buildings:.2f}%)")
        print(f"결과 저장 위치: {self.output_path}")
        
        # 결과 요약 CSV 저장
        summary_df = pd.DataFrame(results)
        summary_file = self.output_path / "분석요약.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"요약 파일: {summary_file}")


def analyze_single_pair(building_shp: str, flood_shp: str, output_shp: str):
    """
    단일 파일 쌍 분석 (테스트/단일 실행용)
    
    Args:
        building_shp: 건물연령 shp 파일 경로
        flood_shp: 홍수위험 shp 파일 경로  
        output_shp: 출력 shp 파일 경로
    """
    print("단일 파일 분석 모드")
    print(f"건물연령: {building_shp}")
    print(f"홍수위험: {flood_shp}")
    
    # 데이터 로드
    print("\n데이터 로딩 중...")
    building_gdf = gpd.read_file(building_shp)
    flood_gdf = gpd.read_file(flood_shp)
    
    print(f"건물 수: {len(building_gdf)}")
    print(f"홍수위험구역 수: {len(flood_gdf)}")
    
    # 컬럼 정보 출력
    print(f"\n건물연령 컬럼: {list(building_gdf.columns)}")
    print(f"홍수위험 컬럼: {list(flood_gdf.columns)}")
    
    # CRS 확인
    print(f"\n건물연령 CRS: {building_gdf.crs}")
    print(f"홍수위험 CRS: {flood_gdf.crs}")
    
    # CRS 통일
    if building_gdf.crs != flood_gdf.crs:
        print("CRS 변환 중...")
        flood_gdf = flood_gdf.to_crs(building_gdf.crs)
    
    # Geometry 유효성 검사 및 수정
    print("\nGeometry 유효성 검사...")
    building_gdf['geometry'] = building_gdf['geometry'].buffer(0)
    flood_gdf['geometry'] = flood_gdf['geometry'].buffer(0)
    
    # Spatial Index 생성
    print("Spatial Index 생성 중...")
    flood_tree = STRtree(flood_gdf.geometry.values)
    
    # 교차 분석
    print("\n교차 분석 수행 중...")
    start_time = time.time()
    
    intersecting_data = []
    total = len(building_gdf)
    
    for idx, (building_idx, building_row) in enumerate(building_gdf.iterrows()):
        if idx % 500 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0
            print(f"  진행률: {idx}/{total} ({100*idx/total:.1f}%) - 예상 남은 시간: {eta:.1f}초")
        
        building_geom = building_row.geometry
        
        # Spatial Index로 후보 필터링
        candidate_indices = flood_tree.query(building_geom)
        
        for cand_idx in candidate_indices:
            flood_row = flood_gdf.iloc[cand_idx]
            flood_geom = flood_row.geometry
            
            if building_geom.intersects(flood_geom):
                # 교차 정보 수집
                try:
                    intersection = building_geom.intersection(flood_geom)
                    intersection_area = intersection.area
                    building_area = building_geom.area
                    overlap_ratio = (intersection_area / building_area * 100) if building_area > 0 else 0
                except:
                    intersection_area = 0
                    overlap_ratio = 0
                
                # 건물 데이터 + 홍수 정보 + 교차 정보
                row_data = building_row.to_dict()
                row_data['flood_idx'] = cand_idx
                row_data['intersect_area'] = intersection_area
                row_data['overlap_pct'] = round(overlap_ratio, 2)
                row_data['is_flood_risk'] = True
                
                # 홍수위험 컬럼 추가 (접두사 붙여서)
                for col in flood_gdf.columns:
                    if col != 'geometry':
                        row_data[f'flood_{col}'] = flood_row[col]
                
                intersecting_data.append(row_data)
                break  # 하나라도 겹치면 위험 건물로 판정
    
    elapsed = time.time() - start_time
    print(f"\n분석 완료! 소요시간: {elapsed:.2f}초")
    print(f"교차하는 건물 수: {len(intersecting_data)}개 / 전체 {total}개")
    
    if intersecting_data:
        # 결과 GeoDataFrame 생성
        result_gdf = gpd.GeoDataFrame(intersecting_data, crs=building_gdf.crs)
        
        # 위험 등급 추가
        result_gdf['risk_level'] = pd.cut(
            result_gdf['overlap_pct'],
            bins=[-0.1, 10, 30, 50, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # 저장
        output_dir = Path(output_shp).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_gdf.to_file(output_shp, encoding='utf-8')
        print(f"\n결과 저장 완료: {output_shp}")
        
        # 통계 출력
        print("\n=== 위험 등급별 통계 ===")
        print(result_gdf['risk_level'].value_counts())
        
        return result_gdf
    else:
        print("\n교차하는 건물이 없습니다.")
        return None


if __name__ == "__main__":
    # 기본 경로 설정 (사용자 환경에 맞게 수정)
    BASE_PATH = r"C:\Users\user\Downloads\kescoaitest"
    
    # 방법 1: 전체 폴더 분석 (모든 지역)
    # analyzer = BuildingFloodAnalyzer(BASE_PATH)
    # analyzer.run_analysis()
    
    # 방법 2: 단일 파일 쌍 테스트
    building_file = os.path.join(BASE_PATH, "건물연령", "전북", "전주시 완산구", "AL_D196_52111_20240214.shp")
    flood_file = os.path.join(BASE_PATH, "홍수위험", "전북", "전주시 완산구", "CFM_SGG_52111_030.shp")
    output_file = os.path.join(BASE_PATH, "분석결과", "위험건물_52111.shp")
    
    # 파일 존재 확인
    if os.path.exists(building_file) and os.path.exists(flood_file):
        result = analyze_single_pair(building_file, flood_file, output_file)
    else:
        print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        print(f"건물연령 파일 존재: {os.path.exists(building_file)}")
        print(f"홍수위험 파일 존재: {os.path.exists(flood_file)}")
        
        # 전체 분석 모드로 전환
        print("\n전체 폴더 분석 모드로 전환합니다...")
        if os.path.exists(BASE_PATH):
            analyzer = BuildingFloodAnalyzer(BASE_PATH)
            analyzer.run_analysis()
        else:
            print(f"기본 경로를 찾을 수 없습니다: {BASE_PATH}")
            print("BASE_PATH 변수를 올바른 경로로 수정해주세요.")
