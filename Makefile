.PHONY: test lint typecheck check bundle dist version clean ui ui-test gateway-test gateway-dev gateway-deploy

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

check: lint typecheck test

# Report viewer (React + Vite, needs Node). Builds ONE self-contained file into
# src/hwb/assets/index.html, which is committed: the rig and `make bundle` never need Node.
ui:
	cd ui && npm ci && npm run build

ui-test:
	cd ui && npm ci && npm run typecheck && npm test

# hwb gateway (Cloudflare Worker): see gateway/README.md.
gateway-test:
	cd gateway && npm ci && npm test

gateway-dev:
	cd gateway && npm ci && (test -f .dev.vars || cp .dev.vars.example .dev.vars) \
		&& npx wrangler d1 migrations apply hwb-reports --local && npm run dev

gateway-deploy:
	cd gateway && npm ci && npm run deploy

# Single-file, dependency-free executable: scp dist/hwb.pyz to the rig and run `python3 hwb.pyz detect`.
# Built from a staging copy so __pycache__ from local test runs never ends up in the bundle.
bundle:
	rm -rf build/pyz dist/hwb.pyz
	mkdir -p build/pyz dist
	cp -R src/hwb build/pyz/
	find build/pyz -name __pycache__ -prune -exec rm -rf {} +
	python3 -m zipapp build/pyz -m "hwb.cli:entry" -p "/usr/bin/env python3" -c -o dist/hwb.pyz
	@echo "built dist/hwb.pyz"

# Everything a release ships: pyz bundle, wheel, sdist, checksums.
dist: bundle
	uv build --out-dir dist
	cp install.sh dist/install.sh
	cd dist && sha256sum hwb.pyz install.sh *.whl *.tar.gz > SHA256SUMS 2>/dev/null || shasum -a 256 hwb.pyz install.sh *.whl *.tar.gz > SHA256SUMS
	@cat dist/SHA256SUMS

version:
	@sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml

clean:
	rm -rf build dist hwb-runs .pytest_cache .mypy_cache .ruff_cache ui/node_modules gateway/node_modules gateway/.wrangler
