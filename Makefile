# Set default goal to help
.DEFAULT_GOAL := help

.PHONY: build-all
build-all: clean-all ## Cleans and then Builds h2o-layoutlm module
	@echo "----- Building h2o-layoutlm module -----"
	$(MAKE) build-h2o-layoutlm

.PHONY: clean-all
clean-all: ## Cleans h2o-layoutlm module
	@echo "----- Clean all h2o-layoutlm module -----"
	$(MAKE) clean-h2o-layoutlm

.PHONY: build-h2o-layoutlm
build-h2o-layoutlm: ## Builds h2o-layoutlm module
	@echo "----- Building h2o-layoutlm module -----"
	pipenv update --dev && \
	(pipenv-setup check || true) && \
		pipenv-setup sync -p && \
		python setup.py bdist_wheel

.PHONY: clean-h2o-layoutlm
clean-h2o-layoutlm: ## Cleans h2o-layoutlm module
	@echo "----- Cleaning h2o-layoutlm module -----"
	rm -rf build dist h2o_layoutlm.egg-info

.PHONY: docker-build-in-docker
docker-build-in-docker: ## pass 'make_target=target' argument to make 'target' in docker container
	@echo "----- Building specified make target in docker container -----"
	docker build \
		-t h2oai/h2oocr-build:0.1.0 \
		-f Dockerfile-build .
	docker run \
		-it --rm \
		-u `id -u`:`id -g` \
		-v `pwd`:/ocr-ai \
		-w /ocr-ai \
		-e "HOME=/ocr-ai" \
		--entrypoint /bin/bash \
		h2oai/h2oocr-build:0.1.0 \
		-c 'pip3 install --user pipenv && \
			export PATH="$${HOME}/.local/bin:$${PATH}" && \
			pipenv update --dev && \
			pipenv run make $(make_target)'

.PHONY: help
help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
