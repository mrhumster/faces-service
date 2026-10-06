IMAGE ?= xomrkob/faces-service
TAG ?= latest

.PHONY: build push deploy venv lint lint-strict test

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

dev-venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

lint:
	python3 -m py_compile app/*.py app/routes/*.py main.py

# py_compile only catches syntax: pyflakes finds undefined names on rarely hit
# branches (a NameError in an inference path is a 500 in production).
lint-strict:
	.venv/bin/python -m pyflakes app/*.py app/routes/*.py main.py

test:
	.venv/bin/python -m pytest tests/ -q

build:
	docker build -t $(IMAGE):$(TAG) .

push:
	docker push $(IMAGE):$(TAG)

deploy: build push
	kubectl apply -f deploy/k8s/
	kubectl rollout status deployment/faces-reader -n go-app --timeout=300s