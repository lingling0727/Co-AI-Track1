import numpy as np
import csv

# 1. F4 연산 테이블 정의 (덧셈: XOR, 곱셈: x^2 + x + 1 = 0 기반)
ADD = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
MUL = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]

def dot_f4(v1, v2):
    res = 0
    for a, b in zip(v1, v2):
        res = ADD[res][MUL[a][b]]
    return res

# 2. PG(3, 4)의 85개 점(표준형) 생성
points = []
# 유형 1: [1, a, b, c]
for a in range(4):
    for b in range(4):
        for c in range(4):
            points.append([1, a, b, c])
# 유형 2: [0, 1, a, b]
for a in range(4):
    for b in range(4):
        points.append([0, 1, a, b])
# 유형 3: [0, 0, 1, a]
for a in range(4):
    points.append([0, 0, 1, a])
# 유형 4: [0, 0, 0, 1]
points.append([0, 0, 0, 1])

# 3. points.csv 저장
with open('points.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['p_id', 'v_p'])
    for i, p in enumerate(points):
        # 리스트를 문자열로 변환하여 저장
        writer.writerow([i, ",".join(map(str, p))])

# 4. incidence.npy 생성 및 저장 (85x85)
# 점 P가 초평면 H에 포함되지 않으면 1, 포함되면 0
incidence = np.zeros((85, 85), dtype=np.int8)
for i in range(85): # 초평면 인덱스
    for j in range(85): # 점 인덱스
        if dot_f4(points[i], points[j]) != 0:
            incidence[i, j] = 1

np.save('incidence.npy', incidence)

print(f"총 {len(points)}개의 점 생성 완료.")
print("points.csv 및 incidence.npy 저장 완료.")