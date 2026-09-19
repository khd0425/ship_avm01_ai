# AI 검출 계획 — 클래스, 학습 데이터, 어안 대응 (FR-7)

## 1. 결정 사항

| 항목 | 결정 |
|------|------|
| 단계 | **일반 화각 카메라 영상으로 먼저** 클래스 구분·학습·추론 체계를 만든다. 어안은 그 다음 (§3) |
| 최초 클래스 | 6종: `person, bollard, fender, quay_edge, small_vessel, buoy` ([ai/classes.yaml](../ai/classes.yaml)) |
| PC 추론 | OpenCV DNN + ONNX (CPU), 파이프라인 안에서 비동기 스레드로 실행 (`inference.backend: opencv_dnn`) |
| Jetson 추론 | 동일한 ONNX → TensorRT (FP16). `InferenceEngine`(TensorRT 자리)은 아직 스텁 |
| 학습 | `ai/prepare_dataset.py` → `ai/train.py` (YOLOv8n 미세조정) → ONNX 내보내기. 데이터 출처·라이선스: [data_attribution.md](data_attribution.md) |

## 2. 최초 6종

사양서 FR-7.1은 "6종"이라 쓰면서 8개 후보(사람, 계선주, 방충재, 안벽 경계, 소형 선박, 계류삭, 부표, 등대)를 나열한다. 유효 거리 **0~50 m 접안 작업**과 **공개 데이터 확보 가능성**을 기준으로 다음과 같이 제안한다 (발주처 확정 필요).

| ID | 클래스 | 정의 | 데이터 상황 |
|----|--------|------|-------------|
| 0 | `person` 사람 | 선원·작업자·수중 인원 | 풍부 (COCO, Open Images 등) |
| 1 | `bollard` 계선주 | 안벽의 계선주/계선 bitt | **소규모 공개 데이터**(Roboflow, 내용 확인 필요), 자체 수집 필요 |
| 2 | `fender` 방충재 | 안벽·선체 방충재 | **공개 데이터 없음** → 자체 수집 |
| 3 | `quay_edge` 안벽 경계 | 접안 대상 안벽 벽면/가장자리 | 공개 데이터 거의 없음(LaRS의 static obstacle 마스크 일부 활용 가능) → 자체 수집. 박스보다 분할이 적합할 수 있음 |
| 4 | `small_vessel` 소형 선박 | 예인선·작업선·요트·어선 | 풍부 (COCO boat, SMD, Open Images 등) |
| 5 | `buoy` 부표 | 항로·계류·어망 부표 | 공개 데이터 있음 (SMD, KOLOMVERSE, Roboflow 등) |

**보류(2단계)**: `mooring_line` 계류삭 — 가늘어 박스 검출에 부적합, 공개 데이터 없음(검색 결과 명시), 분할·자체 데이터 필요. `lighthouse` 등대 — 유효거리 50 m를 크게 벗어나 접안 작업 대상이 아님.

## 3. 어안 영상 대응 전략 (일반 화각 이후)

| 방법 | 내용 | 장단점 |
|------|------|--------|
| A. 보정 뷰에 그대로 적용 | 이미 구현된 가상 카메라(전방 뷰, `undistort` LUT)로 직선화한 영상에 일반 화각 모델을 적용 | 재학습 불필요. 가장자리(±60° 밖)는 왜곡·해상도 저하 |
| **B. 다중 가상 핀홀 타일** | 어안을 좌/중/우 등 여러 가상 핀홀 뷰(`ViewMapper`)로 나눠 각각 검출 → 박스를 `ViewMapper`로 어안/수면 좌표에 역투영 | 일반 화각 학습 데이터를 그대로 활용, 시야 전체 커버. 연산량 ×타일 수. **1순위 권장** |
| C. 어안 직접 학습 | 실제 어안 촬영 영상으로 추가 학습(또는 어안 왜곡을 합성하는 증강) | 가장 정확하나 라벨링 비용. 실측 데이터가 쌓인 뒤 |

권고: **B로 시작 → 실제 어안 데이터가 모이면 C로 보완**. 박스→수면 거리는 `ViewMapper::view_pixel_to_ground` (박스 하단 중심)로 구해 FR-7.3의 0~50 m 판정에 사용한다(설치 높이·피치 정확도 필요, `docs/spec_compliance.md` §3).

## 4. 공개 데이터셋 조사

검색 결과에 명시된 내용만 "확인"으로 표기했다. Roboflow 페이지는 자동 조회가 403으로 막혀 **라이선스를 직접 확인해야 한다**. 상용 납품에는 CC BY-NC 등 비상업 라이선스를 쓸 수 없다.

| 데이터셋 | 내용 | 라이선스(근거) | 적합 클래스 | 이번 사용 |
|----------|------|----------------|-------------|-----------|
| [COCO 2017](https://docs.ultralytics.com/datasets/detect/coco) | 80 클래스, person·boat 포함 | 주석 CC BY 4.0(공식 안내, 이미지는 Flickr 개별 라이선스) — 확인 필요 | person, small_vessel | **사용** (학습 2,385장) |
| [Open Images V7](https://docs.ultralytics.com/datasets/detect/open-images-v7) | 601 클래스: Person, Boat, Barge, Canoe, Watercraft, Lighthouse | 이미지 CC BY 2.0 (검색 결과) | person, small_vessel | 미사용(후보). **Buoy·Bollard·Fender·Dock 클래스 없음(클래스 목록 직접 확인)** |
| [Singapore Maritime Dataset](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset) | 온쇼어/온보드/NIR 영상, 선박 6종 + buoy | Roboflow 판본 CC BY 4.0(검색 결과), 원본 조건은 확인 필요 | small_vessel, buoy | 후보 |
| [KOLOMVERSE](https://arxiv.org/pdf/2206.09885) | 한국 해역 4K 약 215만 장, ship·buoy·fishnet buoy·lighthouse·wind farm | 신청 필요(AI-Hub 계열, 조건 확인 필요) | small_vessel, buoy | 후보 (국내 해역, 발주처 자산 가능성) |
| [LaRS](https://lojzezust.github.io/lars-dataset) | 호수·강·바다 4천+ 키프레임, 8 thing(보트·부표·수영자 등) + 3 stuff(하늘·물·정적 장애물) | 확인 필요 | quay_edge(static obstacle), buoy, person | 후보 |
| [SeaDronesSee](https://seadronessee.cs.uni-tuebingen.de/) | 드론 시점: swimmer, floater, boat, buoy, life jacket | CC0 (검색 결과) | — | 드론 시점이라 선상 카메라와 도메인이 다름. 낮은 우선순위 |
| Roboflow Universe bollard 계열 ([예1](https://universe.roboflow.com/project-m4zhi/bollard-8pt2h/dataset/3), [예2](https://universe.roboflow.com/project-60htx/bollard-v2gn5/dataset/1)) | 418장 / 634장 | **확인 필요** (자동 조회 403) | bollard | 후보. **도로용 볼라드일 수 있어 내용 확인 필수** |
| [Maritime YOLO (earsdataset)](https://universe.roboflow.com/earsdataset-vxvbd/maritime-yolo-iqrc2) | 504장: ship·boat·buoy·lighthouse·anchor | 확인 필요 | small_vessel, buoy | 후보 |
| 계류삭·방충재 | — | — | mooring_line, fender | **공개 데이터 없음**(검색 결과에서 "공개된 계류삭 데이터셋 없음" 명시). 자체 수집 |

`ai/prepare_dataset.py --yolo <export> --yolo-map "원본이름:우리이름"`으로 Roboflow(YOLOv8) 내보내기를 그대로 합칠 수 있다(계정·API 키는 사용자가 발급해 내려받아야 함).

## 5. 학습 결과

두 차례 학습했다. 모델은 모두 YOLOv8n(COCO 사전학습에서 미세조정), TITAN RTX, imgsz 640.

| 실행 | 데이터 | 시간 |
|------|--------|------|
| `avm6_n` | COCO 부분집합(person·boat) — 학습 2,385장 | 25 에폭, 8분 |
| **`avm7_n`** | COCO + Roboflow 4종(부표 2, 수영자, 선박 인식) — **학습 5,189장 / 검증 991장** | 40 에폭, 25분 |

`avm7_n` 검증 결과 (991장, 2,245박스):

| 클래스 | 학습 박스 | 검증 mAP50 | mAP50-95 | 비고 |
|--------|----------:|-----------:|---------:|------|
| person | 7,599 | 0.744 | 0.446 | 수영장 영상 포함(도메인 차이) |
| small_vessel | 3,982 | 0.549 | 0.267 | 원거리·소형 객체가 많아 가장 어려움 |
| buoy | 2,839 | 0.980 | 0.791 | 아래 주의 |
| bollard / fender / quay_edge | **0** | — | — | 데이터 없음. 이 세 클래스는 출력되지 않는다 |
| 전체(3클래스) | | 0.757 | 0.501 | |

**해석 시 주의**
- 학습 로그의 `per-class mAP50-95`에서 데이터 없는 클래스가 0.501로 표시되는 것은 도구가 전체 평균으로 채운 값이며 성능이 아니다.
- **검증셋이 학습 데이터와 같은 출처**(Roboflow 데이터셋을 무작위 분할)라 near-duplicate 프레임이 섞여 있을 수 있다. 특히 buoy 0.98은 **낙관적**이다. 실제 접안 환경 성능은 별도의 독립 검증셋(자체 촬영)으로 측정해야 한다.
- `avm6_n`과 `avm7_n`은 검증셋이 달라 mAP를 직접 비교할 수 없다.
- 표준 사진과 라이브 카메라에서의 확인: `avm7_n`은 bus 사진의 사람 4명을 모두 검출(이전 모델 3명), 부표 이미지에서 buoy 검출, 사진 추론 30~75 ms/장(CPU).
- FR-7.4(사람 검출률 ≥80 %, 30 m 이내)의 평가는 아직 못 함: 거리별 독립 검증 데이터셋이 필요(§6).

## 6. 다음 단계

1. **데이터 확보**: bollard·buoy는 Roboflow/SMD/KOLOMVERSE에서, fender·quay_edge는 발주처 항만 협조로 자체 촬영(일반 화각 + 어안 둘 다). 라벨링 도구(CVAT/Label Studio) 사용.
2. **거리별 검증셋**: 사양서 §3.2 표(5/10/20/30/50 m)에 맞춰 사람·계선주를 촬영해 FR-7.3/7.4 리포트를 만든다.
3. **어안 대응**: §3의 B(다중 가상 핀홀 타일)를 파이프라인에 구현.
4. **Jetson**: ONNX → TensorRT FP16 엔진(`trtexec`), `InferenceEngine` 스텁 교체.
5. **라이선스 확정**: 아래.

## 7. 라이선스 유의

- **모델 프레임워크**: `ultralytics`(YOLOv8/YOLO11)는 **AGPL-3.0**이다. 이 PC의 연구·시험에는 문제없으나, 소스·모델을 발주처에 납품하는 제품에 넣으려면 Ultralytics 엔터프라이즈 라이선스가 필요하거나 허용적 라이선스 대안(YOLOX Apache-2.0 — 현재 사전학습 시험에 사용 —, RT-DETR 등)으로 재학습해야 한다. **M3 전에 결정할 것.**
- **데이터**: 각 데이터셋의 라이선스·저작자표시(CC BY) 조건을 납품 문서에 기록. 비상업(NC) 데이터는 학습에 쓰지 않는다.
- 사양서 §13은 소스코드·지식재산권 귀속을 계약서로 미뤘다. 학습 데이터·모델 가중치의 귀속도 함께 정해야 한다.
