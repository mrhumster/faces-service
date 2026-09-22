IMAGE ?= xomrkob/faces-service
TAG ?= latest

.PHONY: build push deploy venv lint

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

lint:
	python3 -m py_compile app/*.py app/routes/*.py main.py

build:
	docker build -t $(IMAGE):$(TAG) .

push:
	docker push $(IMAGE):$(TAG)

deploy: build push
	kubectl apply -f deploy/k8s/
	kubectl rollout status deployment/faces-reader -n go-app --timeout=300s