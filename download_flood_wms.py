import requests
import urllib3
import os

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 설정 정보
API_KEY = "1MR806CI-1MR8-1MR8-1MR8-1MR806CI0H"  # 사용자가 제공한 키
WMS_URL = "http://www.safemap.go.kr/openApiService/wms/getLayerData.do"
LAYER_NAME = "A2SM_FLUDMARKS"  # 침수흔적도 레이어명 (objtId=212 대응)

# 2. 요청 파라미터 설정 (WMS 표준)
# 전국을 한 번에 받으려면 BBOX를 크게 잡아야 하지만, 상세 확인을 위해 강남역 인근 좌표 사용
# 전국 범위 예시: 126.0, 33.0, 130.0, 38.5 (단, 이 경우 width/height를 매우 크게 해야 함)
bbox_gangnam = "127.027, 37.497, 127.030, 37.500"

params = {
    "apikey": API_KEY,
    "service": "WMS",
    "version": "1.1.1",           # 버전 (보통 1.1.1 또는 1.3.0)
    "request": "GetMap",          # 지도 이미지 요청
    "layers": LAYER_NAME,         # 가져올 레이어
    "styles": "",                 # 기본 스타일
    "format": "image/png",        # 이미지 포맷
    "srs": "EPSG:4326",           # 좌표계 (WGS84 경위도)
    "bbox": bbox_gangnam,         # 영역 (minx, miny, maxx, maxy)
    "width": "1024",              # 이미지 가로 크기 (px)
    "height": "1024",             # 이미지 세로 크기 (px)
    "transparent": "TRUE",        # 배경 투명화 여부
    "bgcolor": "0xFFFFFF"         # 배경색
}

# 3. 요청 및 저장
def download_wms_map():
    try:
        print(f"[{LAYER_NAME}] WMS 요청 중...")
        response = requests.get(WMS_URL, params=params, verify=False)

        # 응답 코드가 200(성공)인지 확인
        if response.status_code == 200:
            # 내용이 에러 메시지(XML/TEXT)인지 이미지인지 확인 필요
            content_type = response.headers.get('Content-Type')

            if 'image' in content_type:
                file_name = "flood_trace_map.png"
                with open(file_name, "wb") as f:
                    f.write(response.content)
                print(f"성공! 이미지가 '{file_name}'으로 저장되었습니다.")
            else:
                print("실패: 이미지가 아닌 데이터가 반환되었습니다.")
                print("응답 내용:", response.text[:500]) # 에러 메시지 출력
        else:
            print(f"HTTP 에러 발생: {response.status_code}")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    download_wms_map()
