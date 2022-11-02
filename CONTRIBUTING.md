# Contributing

Go for it.

## Updating the PYPI package

### 1. Update the version number in setup.cfg

Increment `version` under `metadata` in `setup.cfg`.

Use semantic versioning. See https://semver.org/

### 2. Install the package locally

Use your favored environment manager.

```bash
pip install -e .[dev]
```

### 3. Rerun tests if necessary

```bash
pytest
```

### 4. Check/update MANIFEST.in

Will also check the package configuration.

```bash
check-manifest -u
```

### 5. Build the package

```bash
python -m build
```

### 6. Check the README

```bash
twine check dist/*
```

### 7. Have a PyPI account

If needed, create an account on https://pypi.org/ with an
accessible password.

### 8. Optionally, run against test.pypi.org

Note that https://test.pypi.org requires a separate account
registration.

```bash
twine upload --repository testpypi dist/*
```

And try installing it with,

```bash
pip uninstall -y prompt-hyperopt
pip install --index-url https://test.pypi.org/simple/ prompt-hyperopt
python -m prompt-hyperopt
```

### 9. Upload the package

```bash
twine upload dist/* --verbose
```

### 10. Use the package

```bash
pip install --upgrade prompt-hyperopt
python -m prompt-hyperopt
```
