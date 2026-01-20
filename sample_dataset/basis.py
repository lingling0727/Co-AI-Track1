import numpy as np
import csv
import json

# F4 연산 정의
ADD = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
MUL = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]

def dot_f4(v1, v2):
    res = 0
    for a, b in zip(v1, v2):
        res = ADD[res][MUL[a][b]]
    return res

# 1. PG(2, 4)의 21개 점 생성 (k=3 기초 공간)
base_points = []
for a in range(4):
    for b in range(4):
        base_points.append([1, a, b])
for a in range(4):
    base_points.append([0, 1, a])
base_points.append([0, 0, 1]) # 총 21개

# 2. 74개의 열(column) 구성 (기저 부호의 n=74를 맞춤)
# 실제 연구 데이터에 따라 각 점의 중복도를 할당해야 함.
# 여기서는 예시로 74개를 순환하며 채움.
base_columns = []
for i in range(74):
    base_columns.append(base_points[i % 21])

base_matrix = np.array(base_columns).T # 3 x 74 행렬 생성

# 3. base_matrix.csv 저장
with open('base_matrix.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for row in base_matrix:
        writer.writerow(row)

# 4. 모든 부호어의 무게(w_base) 계산
weights = []
# 모든 0이 아닌 메시지 벡터 m (4^3 - 1 = 63개)
for i in range(4):
    for j in range(4):
        for l in range(4):
            if i == 0 and j == 0 and l == 0: continue
            msg = [i, j, l]
            # 무게 계산: n - (msg와 내적이 0인 열의 개수)
            zero_count = 0
            for col in base_columns:
                if dot_f4(msg, col) == 0:
                    zero_count += 1
            weights.append(74 - zero_count)

# 5. base_info.json 저장
base_info = {
    "n_base": 74,
    "k_base": 3,
    "weights": weights
}
with open('base_info.json', 'w') as f:
    json.dump(base_info, f)

print("base_matrix.csv 및 base_info.json 생성 완료.")