

.default: run-all-tests

PLATFORM := `uname -o`
PROJECT := "sparv_bert_neighbour"

ifeq (${VIRTUAL_ENV},)
  VENV_NAME = .venv
  INVENV = rye run
else
  VENV_NAME = ${VIRTUAL_ENV}
  INVENV =
endif

info:
	@echo "Platform: ${PLATFORM}"
	@echo "INVENV: '${INVENV}'"

dev: install-dev

# setup development environment
install-dev:
	rye sync --no-lock

test: run-all-tests

# run all tests
run-all-tests:
	${INVENV} pytest -vv tests

# run all doc tests
run-doc-tests:
	${INVENV} python -m doctest -v karp_lex/value_objects/unique_id.py

# run all tests with coverage collection
test-w-coverage:
	${INVENV} pytest -vv --cov=src/${PROJECT}  --cov-report=xml tests

# check types
type-check:
	${INVENV} mypy src

# lint the code
lint:
	${INVENV} ruff src tests

part := "patch"
bumpversion: install-dev
	${INVENV} bump2version ${part}

# run formatter(s)
fmt:
	${INVENV} black src tests

# check formatting
check-fmt:
	${INVENV} black --check src tests
