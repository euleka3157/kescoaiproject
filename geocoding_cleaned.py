"""
V-World API 지오코딩 - 정제된 주소 사용
- 입력: 전기화재_3개년_주소정제.csv
- 지번전체주소 → 도로명전체주소 순으로 시도
- 결과: 반지름 6m 원(Polygon)으로 저장
- 중간 저장 기능으로 재개 가능
"""

import pandas as pd
import requests
import urllib3
import time
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import re
import sys

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 설정
# ==========================================
API_KEY = "60099EA2-ECE4-3BA3-9585-5A829AACC420"
INPUT_FILE = Path(r"C:\Users\user\Downloads\kescoaitest\전기화재_3개년_주소정제.csv")
OUTPUT_DIR = Path(r"C:\Users\user\Downloads\kescoaitest\분석결과")

API_URL = "https://api.vworld.kr/req/address"
MAX_RETRIES = 3
RETRY_DELAY = 0.5
REQUEST_DELAY = 0.05
SAVE_INTERVAL = 500
BUFFER_RADIUS = 10  # 반지름 10m

# 시도명 변환
SIDO_MAP = {
    '경기': '경기도', '강원': '강원도', '충북': '충청북도', '충남': '충청남도',
    '전북': '전라북도', '전남': '전라남도', '경북': '경상북도', '경남': '경상남도',
    '제주': '제주특별자치도', '세종': '세종특별자치시',
    '서울': '서울특별시', '부산': '부산광역시', '대구': '대구광역시',
    '인천': '인천광역시', '광주': '광주광역시', '대전': '대전광역시', '울산': '울산광역시',
}


def get_output_paths(year):
    """연도별 출력 파일 경로 생성"""
    return {
        'progress': OUTPUT_DIR / f"지오코딩_{year}_진행상황.csv",
        'success_csv': OUTPUT_DIR / f"전기화재_{year}_좌표변환_5186.csv",
        'success_shp': OUTPUT_DIR / f"전기화재_{year}_좌표변환_5186.shp",
        'invalid_addr': OUTPUT_DIR / f"전기화재_{year}_불확실주소.csv",
        'failed': OUTPUT_DIR / f"전기화재_{year}_변환실패.csv",
    }


def normalize_sido(address):
    """시도명 정규화"""
    if pd.isna(address) or str(address).strip() == '':
        return ''
    addr = str(address).strip()
    for short, full in SIDO_MAP.items():
        if addr.startswith(short + ' '):
            return addr.replace(short + ' ', full + ' ', 1)
    return addr


def check_address_quality(jibun_addr):
    """주소 품질 검사 - 지번주소에 번지가 있는지 확인"""
    if not pd.notna(jibun_addr) or str(jibun_addr).strip() == '':
        return False, '빈주소'

    jibun = str(jibun_addr).strip()

    # 지번주소에 번지가 있는지 확인
    has_jibun = bool(re.search(r'\d+(-\d+)?$', jibun))
    if has_jibun:
        return True, '유효'
    return False, '번지없음'


def get_coordinate(jibun_addr, retries=MAX_RETRIES):
    """V-World API로 좌표 조회 (지번주소만 사용)"""

    addresses_to_try = []

    # 지번주소만 시도
    if pd.notna(jibun_addr) and str(jibun_addr).strip():
        jibun = str(jibun_addr).strip()
        addresses_to_try.append((normalize_sido(jibun), "PARCEL"))
        addresses_to_try.append((jibun, "PARCEL"))

    if not addresses_to_try:
        return None, None, "빈주소"

    for addr, addr_type in addresses_to_try:
        if not addr:
            continue

        params = {
            "service": "address",
            "request": "getcoord",
            "crs": "epsg:5186",
            "address": addr,
            "format": "json",
            "type": addr_type,
            "key": API_KEY
        }

        for attempt in range(retries):
            try:
                response = requests.get(API_URL, params=params, timeout=10, verify=False)
                if response.status_code == 200:
                    json_data = response.json()
                    status = json_data.get('response', {}).get('status', '')
                    if status == 'OK':
                        point = json_data['response']['result']['point']
                        return float(point['x']), float(point['y']), f"성공({addr_type})"
                    elif status == 'NOT_FOUND':
                        break
                    elif status == 'ERROR':
                        return None, None, "API한도초과"
                    else:
                        break
                else:
                    if attempt < retries - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    break
            except Exception:
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                break

    return None, None, "주소없음"


def save_progress(df, progress_path):
    """중간 결과 저장"""
    df.to_csv(progress_path, index=False, encoding='utf-8-sig')


def save_invalid_addresses(df_invalid, output_path):
    """불확실한 주소 목록 저장"""
    if len(df_invalid) > 0:
        df_invalid.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  불확실주소 저장: {output_path} ({len(df_invalid):,}건)")


def save_failed_addresses(df, output_path):
    """변환 실패 목록 저장 (유효한 주소인데 실패한 것)"""
    df_failed = df[(df['지오코딩결과'].notna()) & (df['X_5186'].isna())].copy()
    df_failed = df_failed[~df_failed['지오코딩결과'].str.contains('API한도초과', na=False)]

    if len(df_failed) > 0:
        df_failed.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  변환실패 저장: {output_path} ({len(df_failed):,}건)")


def save_final_results(df, paths):
    """최종 결과 저장 (CSV + SHP) - 성공한 것만, 반지름 6m 원으로"""
    df_success = df[df['X_5186'].notna()].copy()

    if len(df_success) == 0:
        print("  저장할 성공 데이터가 없습니다.")
        return 0

    # CSV 저장
    df_success.to_csv(paths['success_csv'], index=False, encoding='utf-8-sig')
    print(f"  성공 CSV 저장: {paths['success_csv']} ({len(df_success):,}건)")

    # SHP 저장 (반지름 6m 원으로)
    df_success['X_5186'] = df_success['X_5186'].astype(float)
    df_success['Y_5186'] = df_success['Y_5186'].astype(float)

    # Point 생성 후 buffer(6)으로 반지름 6m 원 생성
    geometry = [Point(xy).buffer(BUFFER_RADIUS) for xy in zip(df_success['X_5186'], df_success['Y_5186'])]
    gdf = gpd.GeoDataFrame(df_success, geometry=geometry, crs="EPSG:5186")
    gdf.to_file(paths['success_shp'], driver="ESRI Shapefile", encoding="cp949")
    print(f"  성공 SHP 저장: {paths['success_shp']} ({len(gdf):,}건, 반지름 {BUFFER_RADIUS}m 원)")

    return len(df_success)


def process_year(year):
    """특정 연도 데이터 처리"""
    print("=" * 60)
    print(f"  V-World 지오코딩 - {year}년 데이터")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = get_output_paths(year)

    # 데이터 로딩
    print(f"\n[1단계] 데이터 로딩 및 필터링")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    except:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')

    # 해당 연도만 필터링
    df = df[df['년도'] == year].copy().reset_index(drop=True)
    print(f"  {year}년 데이터: {len(df):,}건")

    # 주소 품질 검사 (지번주소만 확인)
    print(f"\n[2단계] 주소 품질 검사 (지번주소만 사용)")
    quality_results = df['지번전체주소'].apply(check_address_quality)
    df['주소유효'] = quality_results.apply(lambda x: x[0])
    df['주소품질'] = quality_results.apply(lambda x: x[1])

    valid_count = df['주소유효'].sum()
    invalid_count = len(df) - valid_count
    print(f"  ✓ 유효: {valid_count:,}건")
    print(f"  ✗ 불확실: {invalid_count:,}건")

    # 유효한 주소와 불확실한 주소 분리
    df_valid = df[df['주소유효'] == True].copy().reset_index(drop=True)
    df_invalid = df[df['주소유효'] == False].copy()

    # 불확실한 주소 저장
    save_invalid_addresses(df_invalid, paths['invalid_addr'])

    if len(df_valid) == 0:
        print("\n  처리할 유효한 주소가 없습니다.")
        return

    # 기존 진행 상황 확인
    start_idx = 0
    if paths['progress'].exists():
        print(f"\n[기존 진행 파일 발견] {paths['progress']}")
        df_progress = pd.read_csv(paths['progress'], encoding='utf-8-sig')

        processed = df_progress['지오코딩결과'].notna().sum()
        success = df_progress['X_5186'].notna().sum()
        print(f"  이미 처리됨: {processed:,}건 (성공: {success:,}건)")

        if '지오코딩결과' not in df_valid.columns:
            df_valid['X_5186'] = None
            df_valid['Y_5186'] = None
            df_valid['지오코딩결과'] = None

        for col in ['X_5186', 'Y_5186', '지오코딩결과']:
            if col in df_progress.columns:
                df_valid[col] = df_progress[col]

        not_processed = df_valid[df_valid['지오코딩결과'].isna()].index
        if len(not_processed) > 0:
            start_idx = not_processed[0]
            print(f"  재개 지점: {start_idx}번째 행부터")
        else:
            print("  모든 데이터가 이미 처리되었습니다.")
            print(f"\n[최종 저장]")
            save_final_results(df_valid, paths)
            save_failed_addresses(df_valid, paths['failed'])
            return
    else:
        df_valid['X_5186'] = None
        df_valid['Y_5186'] = None
        df_valid['지오코딩결과'] = None

    # API 상태 확인
    print(f"\n[3단계] API 상태 확인")
    test_x, test_y, test_status = get_coordinate("서울특별시 종로구 세종로 1")
    if test_x is None:
        print(f"  API 상태: {test_status}")
        if "한도" in test_status or "ERROR" in test_status:
            print("  API 일일 한도에 도달했습니다. 내일 다시 시도하세요.")
            save_progress(df_valid, paths['progress'])
            save_final_results(df_valid, paths)
            save_failed_addresses(df_valid, paths['failed'])
            return
    else:
        print(f"  API 정상 작동")

    # 지오코딩 시작
    total = len(df_valid)
    remaining = total - start_idx
    print(f"\n[4단계] 지오코딩 진행 ({start_idx}번부터)")
    print(f"  남은 건수: {remaining:,}건")

    success_count = df_valid['X_5186'].notna().sum()
    fail_count = df_valid['지오코딩결과'].notna().sum() - success_count
    start_time = time.time()
    last_save_count = 0
    api_error_count = 0

    for idx in range(start_idx, total):
        jibun_addr = df_valid.at[idx, '지번전체주소']

        time.sleep(REQUEST_DELAY)
        x, y, status = get_coordinate(jibun_addr)

        df_valid.at[idx, 'X_5186'] = x
        df_valid.at[idx, 'Y_5186'] = y
        df_valid.at[idx, '지오코딩결과'] = status

        if x is not None:
            success_count += 1
            api_error_count = 0
        else:
            fail_count += 1
            if "한도" in status:
                api_error_count += 1

        # API 한도 도달 감지
        if api_error_count >= 100:
            print(f"\n  API 한도 도달 감지! 중단합니다.")
            save_progress(df_valid, paths['progress'])
            save_final_results(df_valid, paths)
            save_failed_addresses(df_valid, paths['failed'])
            print(f"\n  내일 다시 실행하면 {idx+1}번째부터 재개됩니다.")
            return

        processed = idx - start_idx + 1

        # 진행률 표시 (100개마다)
        if processed % 100 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining_time = (remaining - processed) / rate if rate > 0 else 0
            print(f"  진행: {idx+1:,}/{total:,} ({100*(idx+1)/total:.1f}%) | "
                  f"성공: {success_count:,} | 실패: {fail_count:,} | "
                  f"속도: {rate:.1f}건/초 | 남은시간: {remaining_time:.0f}초")

        # 중간 저장 (500개마다)
        if processed - last_save_count >= SAVE_INTERVAL:
            save_progress(df_valid, paths['progress'])
            last_save_count = processed
            print(f"    [중간 저장 완료]")

    # 최종 저장
    print(f"\n[5단계] 최종 저장")
    save_progress(df_valid, paths['progress'])
    final_success = save_final_results(df_valid, paths)
    save_failed_addresses(df_valid, paths['failed'])

    # 결과 요약
    print(f"\n" + "=" * 60)
    print(f"  {year}년 처리 완료!")
    print(f"  ─────────────────────────────────")
    print(f"  원본 데이터: {len(df):,}건")
    print(f"  불확실 주소: {len(df_invalid):,}건 (별도 저장)")
    print(f"  처리 대상: {total:,}건")
    print(f"  ─────────────────────────────────")
    print(f"  성공: {success_count:,}건 ({100*success_count/total:.1f}%)")
    print(f"  실패: {fail_count:,}건 ({100*fail_count/total:.1f}%)")
    print(f"  ─────────────────────────────────")
    print(f"  SHP 형태: 반지름 {BUFFER_RADIUS}m 원 (Polygon)")
    print("=" * 60)


def main():
    print("\n" + "=" * 60)
    print("  전기화재 지오코딩 - 정제된 주소 사용")
    print("=" * 60)
    print("\n처리할 연도를 선택하세요:")
    print("  1. 2024년")
    print("  2. 2023년")
    print("  3. 2022년")
    print("  4. 전체 (2022~2024년)")
    print("  q. 종료")

    choice = input("\n선택: ").strip()

    if choice == '1':
        process_year(2024)
    elif choice == '2':
        process_year(2023)
    elif choice == '3':
        process_year(2022)
    elif choice == '4':
        for year in [2024, 2023, 2022]:
            process_year(year)
            print("\n" + "-" * 60 + "\n")
    elif choice.lower() == 'q':
        print("종료합니다.")
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    # 명령줄 인자로 연도 지정 가능
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
            if year in [2022, 2023, 2024]:
                process_year(year)
            else:
                print(f"지원하지 않는 연도: {year}")
        except ValueError:
            print("연도는 숫자로 입력하세요.")
    else:
        main()
