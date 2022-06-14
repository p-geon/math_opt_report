export CONTAINER_NAME = math_opt
export DOCKERFILE = Dockerfile


# utils
.PHONY: build
build: ## docker build
	docker build -f $(DOCKERFILE) -t $(CONTAINER_NAME) .


# -----------
# run-variant
# -----------
void: ## run with shell
	@make build
	docker run -it --rm  \
		-v `pwd`:/work \
		-v `pwd`/$(DIR_DATA):/data \
		$(CONTAINER_NAME) \
		/bin/bash


.PHONY: run
run: ## run normally (with GPUs)
	@make build
	docker run -it --rm --gpus all \
		-v `pwd`:/work \
		-v `pwd`/$(DIR_DATA):/data \
		$(CONTAINER_NAME) \
		python scripts/main.py

# -----------
# utils
# -----------
.PHONY:	help
help:	## show help (this)
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'