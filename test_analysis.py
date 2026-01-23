import sys
sys.path.insert(0, r'C:\Users\user\Downloads\kescoaitest')
import geopandas as gpd
import pandas as pd
from building_multi_risk_analyzer import BuildingMultiRiskAnalyzer

# 결과를 파일로 저장
output_file = r'C:\Users\user\Downloads\kescoaitest\test_result.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    analyzer = BuildingMultiRiskAnalyzer(r'C:\Users\user\Downloads\kescoaitest')

    # 전주시 완산구 분석
    f.write('='*70 + '\n')
    f.write('전주시 완산구 분석\n')
    f.write('='*70 + '\n')
    result1 = analyzer.analyze_region('전북', '전주시 완산구', min_age=None)

    if result1 is not None:
        f.write(f'\n총 건물 수: {len(result1)}\n')
        f.write(f'\n위험코드 분포:\n')
        f.write(str(result1['위험코드'].value_counts()) + '\n')
        f.write(f'\n종합등급 분포:\n')
        f.write(str(result1['종합등급'].value_counts()) + '\n')

        # D, E 등급 각각 확인
        d_count = len(result1[result1['위험코드'] == 'D'])
        e_count = len(result1[result1['위험코드'] == 'E'])
        f.write(f'\nD등급: {d_count}개\n')
        f.write(f'E등급: {e_count}개\n')
        f.write(f'D+E 합계: {d_count + e_count}개\n')

        # 종합점수 분포 확인
        f.write(f'\n종합점수 분포:\n')
        f.write(str(result1['종합점수'].value_counts().sort_index()) + '\n')

    f.write('\n\n')
    f.write('='*70 + '\n')
    f.write('전주시 덕진구 분석\n')
    f.write('='*70 + '\n')
    result2 = analyzer.analyze_region('전북', '전주시 덕진구', min_age=None)

    if result2 is not None:
        f.write(f'\n총 건물 수: {len(result2)}\n')
        f.write(f'\n위험코드 분포:\n')
        f.write(str(result2['위험코드'].value_counts()) + '\n')
        f.write(f'\n종합등급 분포:\n')
        f.write(str(result2['종합등급'].value_counts()) + '\n')

        # D, E 등급 각각 확인
        d_count = len(result2[result2['위험코드'] == 'D'])
        e_count = len(result2[result2['위험코드'] == 'E'])
        f.write(f'\nD등급: {d_count}개\n')
        f.write(f'E등급: {e_count}개\n')
        f.write(f'D+E 합계: {d_count + e_count}개\n')

        # 종합점수 분포 확인
        f.write(f'\n종합점수 분포:\n')
        f.write(str(result2['종합점수'].value_counts().sort_index()) + '\n')

    f.write('\n분석 완료!\n')

print(f'결과가 {output_file}에 저장되었습니다.')
