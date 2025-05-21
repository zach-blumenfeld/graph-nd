rm -rf graph_nd.egg-info
rm -rf dist
poetry build
# You should export this before running the script: export PYPI_TOKEN=your_token
poetry config pypi-token.pypi $PYPI_TOKEN
poetry publish

