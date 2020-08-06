VERSION = $(strip $(shell cat version))
REPO_NAME = $(notdir $(shell pwd))
CONTAINER_NAME = airflow

docker-build:
		docker-compose down
		docker-compose up --build

docker-sh:
		docker exec -it $(CONTAINER_NAME) /bin/bash

bump-version:
		@echo "Bump version..."
		@bash ./.makefiles/bump_version.sh

update-changelog:
		@echo "Update CHANGELOG.md..."
		@bash ./.makefiles/update_changelog.sh
		make re-tag

include ./.makefiles/*.mk
