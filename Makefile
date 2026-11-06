PYTHON ?= python3
UV ?= uv
PYTHONPATH := src
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu124
PYPI_INDEX_URL ?=
PIP_INDEX_ARGS := $(if $(PYPI_INDEX_URL),--index-url $(PYPI_INDEX_URL),)
.PHONY: venv install install-cn install-dev compile train smoke test

venv:
	$(UV) venv --python 3.10 .venv

install:
	$(UV) pip install torch==2.6.0 --index-url $(TORCH_INDEX_URL)
	$(UV) pip install -e ".[all]" $(PIP_INDEX_ARGS)

install-cn:
	$(MAKE) install PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

install-dev: install
	$(UV) pip install -e ".[dev]" $(PIP_INDEX_ARGS)

compile:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall src tests

FLOW_DEVICES ?= 4
FLOW_DATA ?= data/bones
FLOW_TEXT_CACHE ?= data/bones-text
FLOW_OUTPUT ?= outputs/bones-flow
FLOW_ARGS := --data "$(FLOW_DATA)" --text-cache "$(FLOW_TEXT_CACHE)" --output "$(FLOW_OUTPUT)"
FLOW_TRAIN := PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m torch.distributed.run --standalone --nproc_per_node=$(FLOW_DEVICES) -m omg.cli.generation.train

train:
	$(FLOW_TRAIN) $(FLOW_ARGS) $(TRAIN_ARGS)

smoke:
	$(FLOW_TRAIN) $(FLOW_ARGS) --stop-after 12 $(TRAIN_ARGS)

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q
