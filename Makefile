# Every entry point. Python comes from .venv; override with PY=/path/to/python.
PY ?= .venv/bin/python
MODEL ?= configs/models/qwen3.5-9b.yaml
SUITE ?= configs/suites/main.yaml

.PHONY: setup test build smoke eval report monitor tunnel tunnel-status untunnel serve-start serve-stop clean

setup:            ## create .venv from env/bench.txt
	uv venv -p 3.11 .venv
	uv pip install -p .venv -r env/bench.txt

build:            ## materialise .build/hypospace = pristine upstream + patches/
	$(PY) tools/bench.py

test:             ## evaluator self-check with scripted fake LLMs, no network
	$(PY) tools/sanity_check.py

smoke:            ## 2 instances x 2 queries per domain against $MODEL; never reportable
	$(PY) tools/run_eval.py --model $(MODEL) --suite configs/suites/smoke.yaml --tag smoke

eval:             ## full protocol: $(SUITE) against $(MODEL); add ARGS="--served-model NAME --tag T"
	$(PY) tools/run_eval.py --model $(MODEL) --suite $(SUITE) $(ARGS)

report:           ## rebuild results/leaderboard.{md,csv} from runs/
	$(PY) tools/report.py

tunnel:           ## SSH tunnels to both vLLM endpoints (18000 = 27B, 18001 = 9B)
	./scripts/tunnel.sh up

tunnel-status:    ## check which endpoints answer
	./scripts/tunnel.sh status

untunnel:
	./scripts/tunnel.sh down

serve-start:      ## start the 9B vLLM server
	./scripts/vllm9b.sh start

serve-stop:       ## stop the 9B vLLM server
	./scripts/vllm9b.sh stop

monitor:          ## GPU + vLLM queue + per-run progress; add WATCH=30 to refresh
	$(PY) tools/monitor.py $(if $(WATCH),--watch $(WATCH))

clean:            ## remove checkpoints and __pycache__, keep runs/
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
	rm -rf runs/*/checkpoints .build
