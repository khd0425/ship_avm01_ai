# Roboflow로 학습 데이터 확보하기

우리에게 부족한 클래스(`bollard, fender, buoy, quay_edge`)는 공개 데이터가 적어서 (1) Roboflow Universe의 공개
데이터셋을 받아 쓰고 (2) 부족한 부분은 직접 촬영·라벨링해야 한다. 이 문서는 그 절차이며, 아래 명령은 이
저장소의 `ai/` 스크립트와 연결된다 (병합 경로는 Roboflow 내보내기와 같은 구조의 시험 데이터로 검증함).

## 1. 준비 (한 번만)

1. <https://roboflow.com> 에서 계정을 만든다 (무료 플랜으로 시작 가능).
2. **API 키**: 워크스페이스 Settings → *Roboflow API* → *Private API Key*. 스크립트로 내려받을 때만 필요하다.
   키는 코드·채팅·커밋에 넣지 말고 환경 변수로만 쓴다 (`export ROBOFLOW_API_KEY=...`). 노출되면 즉시 재발급.
3. 파이썬 환경: `python3 -m venv ~/envs/avm-rf && ~/envs/avm-rf/bin/pip install roboflow pyyaml`

## 2. 데이터셋 찾기 (Universe)

<https://universe.roboflow.com> 검색창에 클래스별로 검색한다.

| 우리 클래스 | 검색어 예 | 주의 |
|-------------|-----------|------|
| bollard | `mooring bollard`, `bollard quay`, `ship bollard`, `dock cleat` | **도로·보행로용 볼라드(차단봉)** 데이터가 많다. 반드시 예시 이미지를 눈으로 확인 |
| fender | `ship fender`, `dock fender`, `boat fender`, `rubber fender` | 데이터가 거의 없다. 없으면 자체 촬영 |
| buoy | `buoy`, `maritime buoy`, `navigation buoy` | 부표 종류(항로/계류/어망)가 섞여 있음 |
| quay_edge | `quay wall`, `dock`, `pier`, `harbor` | 박스보다 분할(폴리곤) 데이터가 많다 |
| small_vessel | `boat`, `vessel`, `ship detection` | 위성·드론 시점은 제외 |

**데이터셋 페이지에서 반드시 확인할 것** (받기 전에):
- **라이선스** (페이지의 License 항목): `CC BY 4.0`은 출처 표시 조건으로 사용 가능, `Public Domain`/`CC0`은 자유,
  **`CC BY-NC`(비상업)은 납품용 학습에 쓰지 않는다.** 라이선스가 없거나 불명확하면 쓰지 않는다.
- **Classes** 목록과 이미지 수, **샘플 이미지** (도메인이 접안 환경과 맞는지: 선상/안벽 시점, 해상도).
- **Version** 번호 (URL의 `/dataset/3` 의 3).
- 라벨 품질: 몇 장 열어 박스가 정확한지 본다.

## 3. 내려받기 — 두 가지 방법

**A. 웹에서 직접 (API 키 불필요, 로그인 필요)**: 데이터셋 페이지 → *Download Dataset* → 형식 **YOLOv8**
→ *download zip to computer* → 압축을 `datasets/raw/<이름>/` 에 푼다.
(`train/images, train/labels, valid/…, test/…, data.yaml` 구조가 나와야 한다.)

**B. 스크립트**:
```bash
export ROBOFLOW_API_KEY=...            # 본인 키
~/envs/avm-rf/bin/python ai/download_roboflow.py \
    --url https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
# -> datasets/raw/<project>/  (+ SOURCE.txt: 출처·날짜. licence 줄은 직접 채운다)
```
`datasets/raw/` 는 `.gitignore` 에 들어 있어 라이선스 확인 전 데이터가 실수로 커밋되지 않는다.

## 4. 우리 6종 데이터에 합치기

내려받은 `data.yaml` 의 `names` 를 열어 클래스 이름(**대소문자 구분**)을 확인하고, 우리 클래스로 매핑한다.
매핑하지 않은 클래스는 버려지고, 매핑된 객체가 하나도 없는 이미지는 건너뛴다.
```bash
~/envs/avm-ai/bin/python ai/prepare_dataset.py --out datasets/avm6 --append \
    --yolo datasets/raw/bollard-8pt2h --yolo-map "Bollard:bollard,mooring bollard:bollard"
# 데이터셋마다 --yolo / --yolo-map 을 한 쌍씩 반복 가능
```
출력의 `boxes per class` 표에서 해당 클래스 수가 늘었는지, `WARNING: no training data yet for:` 목록이 줄었는지
확인한다. 처음부터 다시 만들려면 `--append` 없이 COCO(`--coco-zip`)와 `--yolo` 들을 한 번에 지정한다.

## 5. 내가 찍은 이미지 라벨링

접안 환경(안벽, 방충재, 계선주)은 직접 찍은 데이터가 가장 중요하다.
1. **일반 화각 카메라와 어안 카메라 둘 다** 촬영한다 (거리 5/10/20/30/50 m, 아침·저녁·역광, 사람 포함).
2. Roboflow → *Create New Project* → *Object Detection* → 클래스 이름을 **우리 이름 그대로** 만든다
   (`person, bollard, fender, quay_edge, small_vessel, buoy`).
3. 이미지를 올리고 박스를 그린다 (한 클래스가 여러 개면 전부 그린다).
4. *Generate Version*: 전처리(리사이즈 등)는 최소로 하고 **Roboflow의 증강(augmentation)은 끄는 것**을 권한다
   (학습 때 ultralytics가 증강한다). Train/Valid/Test = 70/20/10.
5. *Export* → **YOLOv8** → 위 3-A와 같이 받아 4번 절차로 합친다 (이름을 우리와 같게 만들었으면 매핑도 같은 이름).

> **주의 — 데이터 공개**: 무료 플랜의 프로젝트는 공개될 수 있는 것으로 알고 있다(요금제 약관을 직접 확인할 것).
> **발주처 항만·선박 영상이나 얼굴이 찍힌 영상은 공개 프로젝트에 올리지 않는다.** 그런 데이터는 로컬 라벨링
> 도구(CVAT, Label Studio 등, 서버에 데이터가 남지 않음)로 라벨링하고 YOLO 형식으로 내보낸다.

## 6. 학습·확인

```bash
~/envs/avm-ai/bin/python ai/train.py --data datasets/avm6/data.yaml --epochs 30 --name avm6_n
./build-pc/detect_image --model models/avm6_n.onnx --arch yolov8 --out out 사진1.jpg 사진2.jpg
```
학습 결과의 클래스별 mAP 표를 보고, 데이터가 적은 클래스(수백 장 미만)는 값이 낮을 것을 감안한다.
`config/avm_pc_demo.json` 의 `inference.model_path` 를 새 모델로 바꾸면 카메라 영상에서 바로 확인된다.

## 7. 기록 (납품·감사 대비)
받은 데이터셋마다 `SOURCE.txt`(URL, 버전, 날짜, **라이선스**)를 채우고, `docs/ai_dataset_plan.md` §4 표를
갱신한다. CC BY 데이터는 출처 표시를 납품 문서에 포함한다.
