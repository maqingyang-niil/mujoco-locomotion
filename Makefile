.PHONY: check train clear evaluate

RUN ?=go2_ppo
PY ?=uv run python -m
CONFIG ?=go2/configs/training_config.yaml
MODEL ?=go2/models/$(RUN)_final.zip
VECNORMALIZE ?= go2/models/$(RUN)_vecnormalize.pkl





check:
	$(PY) go2.scripts.check_env

train:
	$(PY) go2.scripts.train --config $(CONFIG)
	
clean:
	$(PY) go2.scripts.clear

evaluate:
	$(PY) go2.scripts.evaluate --model $(MODEL) --vec-normalize $(VECNORMALIZE) --episodes 10 --steps 3000

evaluate-grid:
	$(PY) go2.scripts.evaluate --model $(MODEL) --vec-normalize $(VECNORMALIZE) --grid

play:
	$(PY) go2.scripts.play --model $(MODEL) --vec-normalize $(VECNORMALIZE)