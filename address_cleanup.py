"""
전기화재_3개년_통합.csv 주소 정제 스크립트
- 지번전체주소: 시도 + 구군 + (구군2) + 읍면리동 + (리명) + 번지
- 도로명전체주소: 시도 + 구군 + (구군2) + (읍면) + 도로명 + (건물명 또는 번지)
"""

import pandas as pd
import re
from pathlib import Path

INPUT_FILE = Path(r"C:\Users\user\Downloads\kescoaitest\전기화재_3개년_통합.csv")
OUTPUT_FILE = Path(r"C:\Users\user\Downloads\kescoaitest\전기화재_3개년_주소정제.csv")


def extract_road_name(ri_val):
    """리 컬럼에서 도로명 추출"""
    if pd.isna(ri_val) or str(ri_val).strip() == '':
        return ''
    ri = str(ri_val).strip()
    match = re.search(r'([가-힣0-9]+(?:로|길|대로)\d*번?길?)', ri)
    if match:
        return match.group(1)
    return ''


def extract_ri_name(ri_val):
    """리 컬럼에서 리명 추출"""
    if pd.isna(ri_val) or str(ri_val).strip() == '':
        return ''
    ri = str(ri_val).strip()
    match = re.search(r'([가-힣]+리)', ri)
    if match:
        return match.group(1)
    return ''


def extract_jibun(jibun_val):
    """지번상세에서 번지만 추출 (불완전번지는 - 제거)"""
    if pd.isna(jibun_val) or str(jibun_val).strip() == '':
        return ''
    jibun = str(jibun_val).strip()
    # 완전번지: 123-45
    match = re.match(r'^(\d+)-(\d+)', jibun)
    if match:
        return f'{match.group(1)}-{match.group(2)}'
    # 불완전번지: 123- 뒤에 숫자 아닌 것 (- 제거)
    match = re.match(r'^(\d+)-(?:\s|[^0-9]|$)', jibun)
    if match:
        return match.group(1)
    # 숫자만: 123 뒤에 공백
    match = re.match(r'^(\d+)\s', jibun)
    if match:
        return match.group(1)
    return ''


def extract_building_name(jibun_val):
    """지번상세에서 건물명/상세 추출 (번지 제외한 나머지)"""
    if pd.isna(jibun_val) or str(jibun_val).strip() == '':
        return ''
    jibun = str(jibun_val).strip()
    # 완전번지 뒤의 내용: 123-45 xxx
    match = re.match(r'^\d+-\d+\s+(.+)$', jibun)
    if match:
        return match.group(1).strip()
    # 불완전번지 뒤의 내용: 123- xxx
    match = re.match(r'^\d+-\s+(.+)$', jibun)
    if match:
        return match.group(1).strip()
    # 숫자만 뒤의 내용: 123 xxx
    match = re.match(r'^\d+\s+(.+)$', jibun)
    if match:
        return match.group(1).strip()
    return ''


def is_eup_myeon(dong_val):
    """읍면동에서 읍/면인지 확인"""
    if pd.isna(dong_val) or str(dong_val).strip() == '':
        return False
    dong = str(dong_val).strip()
    return dong.endswith('읍') or dong.endswith('면')


def safe_str(val):
    """안전하게 문자열 변환 (NaN 처리)"""
    if pd.isna(val):
        return ''
    return str(val).strip()


def build_jibun_address(row):
    """지번전체주소 생성"""
    parts = []

    # 시도
    sido = safe_str(row['시도'])
    if sido:
        parts.append(sido)

    # 구군
    gugun = safe_str(row['구군'])
    if gugun:
        parts.append(gugun)

    # 구군2 (있으면)
    gugun2 = safe_str(row['구군2'])
    if gugun2:
        parts.append(gugun2)

    # 읍면리동
    dong = safe_str(row['읍면리동'])
    if dong:
        parts.append(dong)

    # 리명 (있으면)
    ri_name = extract_ri_name(row['리'])
    if ri_name:
        parts.append(ri_name)

    # 번지
    jibun = extract_jibun(row['지번상세'])
    if jibun:
        parts.append(jibun)

    return ' '.join(parts)


def build_road_address(row):
    """도로명전체주소 생성 (도로명 + 건물명만, 번지는 제외)"""
    # 도로명이 없으면 빈 문자열
    road_name = extract_road_name(row['리'])
    if not road_name:
        return ''

    parts = []

    # 시도
    sido = safe_str(row['시도'])
    if sido:
        parts.append(sido)

    # 구군
    gugun = safe_str(row['구군'])
    if gugun:
        parts.append(gugun)

    # 구군2 (있으면)
    gugun2 = safe_str(row['구군2'])
    if gugun2:
        parts.append(gugun2)

    # 읍면 (읍/면인 경우만)
    dong = safe_str(row['읍면리동'])
    if is_eup_myeon(dong):
        parts.append(dong)

    # 도로명
    parts.append(road_name)

    # 건물명만 (번지는 제외 - 숫자-숫자 형태는 지번이므로)
    building = extract_building_name(row['지번상세'])
    if building:
        parts.append(building)

    return ' '.join(parts)


def main():
    print("=" * 60)
    print("  전기화재 주소 정제")
    print("=" * 60)

    # 데이터 로딩
    print(f"\n[1단계] 데이터 로딩")
    try:
        df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    except:
        df = pd.read_csv(INPUT_FILE, encoding='cp949')

    print(f"  총 건수: {len(df):,}건")

    # 주소 정제
    print(f"\n[2단계] 주소 정제 중...")
    df['지번전체주소'] = df.apply(build_jibun_address, axis=1)
    df['도로명전체주소'] = df.apply(build_road_address, axis=1)

    # 필요한 컬럼만 선택
    result = df[['년도', '순번', '화재발생일시', '지번전체주소', '도로명전체주소']].copy()

    # 결과 저장
    print(f"\n[3단계] 결과 저장")
    result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"  저장 완료: {OUTPUT_FILE}")

    # 결과 요약
    print(f"\n" + "=" * 60)
    print(f"  정제 결과 요약")
    print(f"=" * 60)

    jibun_valid = (result['지번전체주소'] != '').sum()
    road_valid = (result['도로명전체주소'] != '').sum()

    print(f"  총 건수: {len(result):,}건")
    print(f"  지번주소 있음: {jibun_valid:,}건 ({100*jibun_valid/len(result):.1f}%)")
    print(f"  도로명주소 있음: {road_valid:,}건 ({100*road_valid/len(result):.1f}%)")

    # 샘플 출력
    print(f"\n[샘플 10건]")
    for i, row in result.head(10).iterrows():
        print(f"\n{i+1}. 년도: {row['년도']}, 순번: {row['순번']}")
        print(f"   지번: {row['지번전체주소']}")
        print(f"   도로명: {row['도로명전체주소']}")


if __name__ == "__main__":
    main()
