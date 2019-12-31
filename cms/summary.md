| date| name | 알고리즘 | 변수개수 | CV | LB | 비고 |
|-----|------|---------|---------|----|-----|-----|
| 2019-10-18 | 차명섭 | LGB | - | - | 1.29 | Initial Commit |
| 2019-11-01 | 차명섭 | LGB | - | - | 1.12 | Baseline |
| 2019-11-14 | 차명섭 | LGB | - | - | 1.10 | Add Features(time_lag)|
| 2019-11-18 | 차명섭 | LGB | - | -  | 1.08 | Add UCF Features  |
| 2019-11-28 | 차명섭 | LGB | - | -  | 0.99 | Blending for 3 submissions  |
| 2019-12-10 | 차명섭 | LGB | - | -  | 0.97 | Blending & Best parameters  |
| 2019-12-19 | 차명섭 | LGB | - | -  | 0.958 | Blending & Hyperparameter  |

# 2019-10-18 ~ 11-01
- Competition 배경지식 검색 및 공부
- Baseline 모델 탐색
- EDA 코드 정리
- Feature Engineering List 작성

# 2019-11-09 ~ 11-16
- Baseline Model 선정(1.12)
- Feaure Engineering List 구현
  - Time lag, Sliding windows
- Data Preprocessing
  - Data leakage, Missing value


# 2019-11-16 ~ 12-09
  - Data leakage 반영 삭제
  - 파라미터 최적화 (Grid Search)

# 2019-12-10 ~ 16
  - 3개의 커널 값에 대해 Ensemble 모델 구현


# 2019-12-16 ~ 종료
    - Ranking average, weight average, stacking, stacking net, Blending 구현
