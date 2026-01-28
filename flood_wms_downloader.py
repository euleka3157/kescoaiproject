"""
침수흔적도 WMS 다운로드 프로그램
================================
safemap.go.kr에서 침수흔적도 레이어를 다운로드합니다.
SSL 보안 우회 적용됨.

사용법:
    python flood_wms_downloader.py

주의: 유효한 API 키가 필요합니다.
      https://www.safemap.go.kr 에서 회원가입 후 API 키 발급받으세요.
"""

import requests
import urllib3
import os

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============== 설정 ==============
API_KEY = "여기에_발급받은_API키_입력"  # safemap.go.kr에서 발급받은 키
WMS_URL = "https://www.safemap.go.kr/openapi2/IF_0092_WMS"
LAYER_NAME = "A2SM_FLUDMARKS"  # 침수흔적도 레이어

# 저장 폴더
OUTPUT_DIR = "침수흔적도"
# ==================================

# 지역별 BBOX 설정 (EPSG:4326 - WGS84 경위도)
# 형식: minX, minY, maxX, maxY (서, 남, 동, 북)
REGIONS = {
    "서울전체": "126.764, 37.428, 127.183, 37.701",
    "강남역": "127.024, 37.495, 127.032, 37.502",
    "부산전체": "128.850, 35.050, 129.250, 35.350",
    "대구전체": "128.400, 35.750, 128.750, 36.000",
    "인천전체": "126.550, 37.350, 126.800, 37.600",
    "광주전체": "126.750, 35.050, 127.000, 35.250",
    "대전전체": "127.250, 36.250, 127.500, 36.450",
    "전국": "124.5, 33.0, 132.0, 43.0",
}


def download_wms_map(region_name="서울전체", width=2048, height=2048):
    """
    WMS 침수흔적도 다운로드

    Args:
        region_name: REGIONS에 정의된 지역명 또는 직접 bbox 문자열
        width: 이미지 가로 크기 (픽셀)
        height: 이미지 세로 크기 (픽셀)
    """

    # API 키 확인
    if API_KEY == "여기에_발급받은_API키_입력" or not API_KEY:
        print("=" * 50)
        print("오류: API 키가 설정되지 않았습니다!")
        print("1. https://www.safemap.go.kr 접속")
        print("2. 회원가입 및 로그인")
        print("3. 오픈API -> 인증키 발급")
        print("4. 발급받은 키를 이 파일의 API_KEY에 입력")
        print("=" * 50)
        return False

    # BBOX 설정
    if region_name in REGIONS:
        bbox = REGIONS[region_name]
    else:
        bbox = region_name  # 직접 bbox 입력한 경우

    params = {
        "serviceKey": API_KEY,
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": LAYER_NAME,
        "styles": "",
        "format": "image/png",
        "srs": "EPSG:4326",
        "bbox": bbox,
        "width": str(width),
        "height": str(height),
        "transparent": "TRUE"
    }

    print(f"\n[침수흔적도 다운로드]")
    print(f"지역: {region_name}")
    print(f"BBOX: {bbox}")
    print(f"크기: {width}x{height}px")
    print(f"요청 중...")

    try:
        # SSL 검증 우회 (verify=False)
        response = requests.get(
            WMS_URL,
            params=params,
            verify=False,  # SSL 우회
            timeout=60
        )

        print(f"상태코드: {response.status_code}")

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')

            if 'image' in content_type:
                # 저장 폴더 생성
                os.makedirs(OUTPUT_DIR, exist_ok=True)

                # 파일명 생성
                safe_name = region_name.replace(" ", "_").replace(",", "_")
                file_name = os.path.join(OUTPUT_DIR, f"침수흔적도_{safe_name}.png")

                with open(file_name, "wb") as f:
                    f.write(response.content)

                file_size = len(response.content)
                print(f"성공! 저장됨: {file_name}")
                print(f"파일 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                return True
            else:
                print("오류: 이미지가 아닌 응답")
                print(response.text[:500])
                return False
        else:
            print(f"HTTP 에러: {response.status_code}")
            print(response.text[:500])
            return False

    except requests.exceptions.Timeout:
        print("오류: 요청 시간 초과 (60초)")
        return False
    except Exception as e:
        print(f"오류 발생: {e}")
        return False


def download_all_regions():
    """모든 정의된 지역 다운로드"""
    print("=" * 50)
    print("전체 지역 침수흔적도 다운로드 시작")
    print("=" * 50)

    success_count = 0
    for region in REGIONS:
        if download_wms_map(region):
            success_count += 1

    print(f"\n완료: {success_count}/{len(REGIONS)} 지역 다운로드 성공")


if __name__ == "__main__":
    print("=" * 50)
    print("침수흔적도 WMS 다운로드 프로그램")
    print("=" * 50)

    print("\n다운로드 옵션:")
    print("1. 서울 전체")
    print("2. 부산 전체")
    print("3. 전국")
    print("4. 모든 지역")
    print("5. 직접 입력 (bbox)")

    choice = input("\n선택 (1-5): ").strip()

    if choice == "1":
        download_wms_map("서울전체")
    elif choice == "2":
        download_wms_map("부산전체")
    elif choice == "3":
        download_wms_map("전국", width=4096, height=4096)
    elif choice == "4":
        download_all_regions()
    elif choice == "5":
        bbox = input("BBOX 입력 (minX,minY,maxX,maxY): ").strip()
        download_wms_map(bbox)
    else:
        print("서울전체로 다운로드합니다.")
        download_wms_map("서울전체")
