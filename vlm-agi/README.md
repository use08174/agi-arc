# VLM ARC Refactor

`vlm-test (5).ipynb`를 실행용 파이썬 코드로 옮긴 디렉터리입니다.

## Files

- `main.py`: 실행 진입점
- `config.py`: 환경변수 기반 설정 로딩
- `runtime.py`: Kaggle/로컬 런타임 준비
- `model.py`: Qwen3-VL 로딩 및 생성
- `grid.py`: ARC 그리드 변환/시각화 유틸
- `prompts.py`: scene/policy 프롬프트 빌더
- `session.py`: ARC 세션과 multi-action 루프
- `scene_probe.py`: 장면 이해 probe 루프

## Usage

```bash
python vlm-agi/main.py --game-id su15 --max-steps 30
```

장면 이해 probe를 돌리려면:

```bash
python vlm-agi/main.py --scene-probe --probe-steps 5
```

필요한 값은 환경변수로 넣습니다.

- `ARC_API_KEY`
- `LOCAL_VLM_MODEL_PATH`
- `ARC_MODE`
- `ARC_GAME_ID`
