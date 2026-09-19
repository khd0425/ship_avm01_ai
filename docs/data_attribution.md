# 학습 데이터 출처·라이선스 기록

CC BY 4.0 데이터는 **출처 표시**가 조건이다. 납품 문서·모델 설명에 아래 표를 포함한다. 라이선스는 각 데이터셋의
`README.dataset.txt`(Roboflow 내보내기에 포함)에 적힌 값이며, 웹 페이지의 최신 라이선스와 다르지 않은지 납품 전에 한 번 더 확인한다.

| 데이터셋 | 출처 (Roboflow Universe) | 라이선스 | 우리 클래스로의 사용 |
|----------|--------------------------|----------|----------------------|
| buoy | <https://universe.roboflow.com/hyundeok-kang/buoy-cvi08> | CC BY 4.0 | `buoy` (391장 / 407박스) |
| Buoy | <https://universe.roboflow.com/hyundeok-kang/buoy-5l1pv-bsrge> | CC BY 4.0 | `buoy` (2,013장 / 2,704박스) |
| Drowning Detect 2 | <https://universe.roboflow.com/hyundeok-kang/drowning-detect-2-unwcw> | CC BY 4.0 | swimming·drowning·Person out of water → `person` (588장 / 1,629박스) |
| Ship Recognition | <https://universe.roboflow.com/hyundeok-kang/ship-recognition-ohjxv-bq5v3> | Public Domain | boat·fishing boat → `small_vessel`, buoy → `buoy` (542장). 대형선 포함 78장 제외 |
| COCO 2017 | <https://cocodataset.org> | 주석 CC BY 4.0, 이미지는 Flickr 개별 라이선스(확인 필요) | person, boat → `person`, `small_vessel` |

(Roboflow 페이지는 "Provided by a Roboflow user"로 표기하며 원저작자가 다를 수 있다 — 원본 출처를 확인할 수 있으면 함께 기록.)

## 데이터 품질 메모
- **Drowning Detect 2**는 수영장·실내 영상이 많아 해상 환경과 도메인이 다르다. 사람 검출의 다양성에는 도움이 되지만 해상 성능 지표로는 신뢰하지 말 것.
- **buoy(1번)**는 근접 촬영이 많고, 일부 이미지는 화면 속 다른 부표가 라벨링되지 않았다(누락 라벨 → 학습 잡음).
- **Ship Recognition**의 화물선·여객선·군함·항공모함은 `small_vessel`이 아니어서 그런 대상이 있는 이미지는 통째로 제외했다.
- `bollard`, `fender`, `quay_edge`는 이번 데이터에도 없다 (여전히 학습 데이터 0).
