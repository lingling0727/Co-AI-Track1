import pandas as pd
import numpy as np
import json

# 1. 기존 데이터 로드
points_df = pd.read_csv('points.csv')
# base_matrix는 3x74 형태이므로 전치하여 74개의 열 벡터로 취급
base_matrix = pd.read_csv('base_matrix.csv', header=None).values.T

# 점 좌표를 리스트 형태로 변환 (비교용)
all_points = [list(map(int, p.split(','))) for p in points_df['v_p']]

# 2. u_p (하한) 계산: 기저 부호에서 각 점이 몇 번 쓰였는지 카운트
u_p = {}
for i, p in enumerate(all_points):
    # PG(3, 4)의 점은 4차원이지만 기저 점은 3차원이므로 
    # 앞의 3개 성분만 비교하거나 기저 점을 4차원으로 확장해서 비교해야 함
    # 여기서는 기저 점이 PG(3, 4)의 특정 부분집합이라 가정하고 카운트함
    count = 0
    for col in base_matrix:
        # 기저의 3차원 벡터를 [a, b, c, 0] 형태로 간주하여 비교
        if list(col) == p[:3] and p[3] == 0:
            count += 1
    u_p[i] = count

# 3. lambda_p (상한) 계산: 하한에 여유분(Slack)을 더함
# 목표 길이 n=76, 기저 n=74이므로 추가할 수 있는 열은 단 2개임
lambda_p = {}
for i in range(len(all_points)):
    # 모든 점에 대해 최대 2번까지 더 선택할 수 있도록 허용 (탐색 범위 확보)
    lambda_p[i] = u_p[i] + 2

# 4. bounds.json 저장
bounds_data = {
    "u_p": u_p,
    "lambda_p": lambda_p
}

with open('bounds.json', 'w') as f:
    json.dump(bounds_data, f)

print("bounds.json 생성 완료.")