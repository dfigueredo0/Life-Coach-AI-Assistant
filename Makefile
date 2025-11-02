setup:
	poerty instgall
lint:
	poerty run ruff check src tests
	poerty run black --check src tests
test:
# --disable-warnings check if needed after running
	poetry run pytest -v --maxfail=1 	
plan:
	poetry run python -m planning.decompse
schedule:
	poetry run python -m schedule.adapters
nudge:
	poetry run python -m nudges.engine
recovery:
	poetry run python -m nudges.recovery
summary:
	poetry run python -m weekly_summary
serve:
	poetry run uvicorn service.app:app --reload