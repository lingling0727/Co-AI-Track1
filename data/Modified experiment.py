def get_sorted_candidates(self, current_solution, candidates):
        """
        후보 점들을 '가장 안전한 순서'로 정렬합니다.
        (Heuristic: Greedy Approach)
        점수 기준: 이 점을 추가했을 때, 가장 붐비는 초평면의 점 개수가 적을수록 좋음.
        """
        scored_candidates = []
        
        # 현재까지의 초평면 점 개수 상태 미리 계산
        current_indices = current_solution
        if not current_indices:
            current_counts = np.zeros(self.incidence.shape[0])
        else:
            sub_incidence = self.incidence[:, current_indices]
            current_counts = np.sum(sub_incidence, axis=1)
            
        for cand_idx in candidates:
            # 이 후보 점이 포함된 초평면들만 카운트 +1 해봄
            # 전체 행렬 연산 대신, 해당 컬럼만 더해서 최대값 예측
            cand_col = self.incidence[:, cand_idx]
            next_counts = current_counts + cand_col
            
            # Max Count가 작을수록 안전함 (여유가 있음)
            # 동점일 경우를 대비해, '꽉 찬 초평면의 개수'를 2순위로 둠
            max_c = np.max(next_counts)
            sum_c = np.sum(next_counts) # 혹은 꽉 찬 평면 개수
            
            scored_candidates.append((max_c, sum_c, cand_idx))
            
        # 오름차순 정렬 (Max Count가 작은 것부터 = 안전한 것부터)
        scored_candidates.sort(key=lambda x: (x[0], x[1]))
        
        # 인덱스만 추출하여 반환
        return [x[2] for x in scored_candidates]

    def backtrack(self, current_solution):
        # 1. 종료 조건
        if self.found_solution is not None: return
        self.nodes_visited += 1
        
        # 2. 유효성 검사
        if not self.is_valid(current_solution): return

        # 3. 성공 조건
        if len(current_solution) == self.n:
            self.found_solution = current_solution
            return

        # 4. LP 가지치기
        if len(current_solution) >= self.k:
            self.lp_calls += 1
            if not solve_lp_relaxation(current_solution, self.incidence, self.n, self.d, self.k, self.q):
                return

        # 5. 다음 점 선택 (Heuristic 적용!)
        last_index = current_solution[-1] if current_solution else -1
        remaining_needed = self.n - len(current_solution)
        
        start_idx = last_index + 1
        end_idx = self.num_points - remaining_needed + 1
        
        # 단순히 range로 돌지 않고, 후보군을 리스트로 만듦
        raw_candidates = list(range(start_idx, end_idx))
        
        # === [핵심] 후보 점들을 스마트하게 정렬 ===
        # 탐색 깊이가 얕을 때는 정렬 비용이 아깝지 않음 (가지치기 효과가 크므로)
        sorted_candidates = self.get_sorted_candidates(current_solution, raw_candidates)
        
        for i in sorted_candidates:
            current_solution.append(i)
            self.backtrack(current_solution)
            if self.found_solution: return
            current_solution.pop()
    run_experiment(n_in, k_in, d_in, q_in)
