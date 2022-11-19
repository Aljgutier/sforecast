
# Python Packaging (Notes)

[Python Packages Book](https://py-pkgs.org/)

by Tomas Beuzen & Tiffany Timbers

Create Virtual Environment
> conda create --name sforecast python=3.8 -y
> conda activate sforecast

Install Poetry Packaging Software 
> conda install -c conda-forge cookiecutter

Create package Structure .. one directory in the ../workespaces directory
> cookiecutter https://github.com/py-pkgs/py-pkgs-cookiecutter.git


Git ... go to git and create repo ... Init, Readme ... all defaults!!!
> echo "# sforecast" >> README.md
> git add .
> git commit -m "feat: check in initial package"
> git branch -M main
> git remote add origin https://github.com/Aljgutier/beautifulplots.git
> git push origin main

Poetry Lock File
> poetry install  # writes the lock file (package dependencies)

Add Source Code
> git add src/pycounts/beautifulplots.py
> git commit -m "feat: kale source code"

Add dependent packages with poetry
> poetry add matplotlib
> poetry add seaborn
> poetry add pandas
> poetry add sklearn
> poetry add tensorflow
> poetry add statsmodels
> poetry add pmdarima


* we recommend specifying version constraints without an upper cap by manually changing poetry’s default caret operator (^) to a greater-than-or-equal-to sign (>=)  
pyproject.toml  
   ```
  [tool.poetry.dependencies]
  python = ">=3.8"
  matplotlib = ">=3.4.3"
  ```

Add Dev Packages with poetry ... Jupyter notebook
> poetry add --dev jupyter
> poetry add --dev beautifulplots
> poetry add --dev xlrd
``

Tests
  * Poetry adds
    * poetry add --dev pytest
    * poetry add --dev pytest-cov # test coverage

  * testing with pytest
    * create test file in tests/test_sforecast.py
    * pytest tests/  # run tests
    * pytest tests/ --cov=sforecast


Documentation

  * Sphinx
  * myst ... Markedly Styled Text
 ```
  > poetry add --dev myst-nb --python "^3.8"
  > poetry add --dev sphinx-autoapi sphinx-rtd-theme

  > cd ./docs
  > make html
  > cd ./_build/html
  > open index.html
 ```

Read the Docs
    * https://readthedocs.org/
    * associate git account with your read the docs account
    * Create a new packag/project ... add the git repo from your linked account
    * Build latest

Tagging Package with Version
   * tag your local branch
    ```
    > update the version in project.toml and commit
    > git tag v0.1.0
    > git push --tags
    > git tag -l # list local tags
    > git tag -d v2.0 # delete local tag
    > git ls-remote --tags origin # list remote tags
    > git push --delete origin v1.0 # delete remote tag
    ```
  * git online package your release and associate with the tag

  ![Git package release](./git_release_package.png)

Building and Distributing Your Package**

  * > poetry build
    * sdist - software distribution
    * wheel - prebuilt distribution

  * pip install wheel
    * > cd dist/
    * >pip install beautifulplots-0.1.1-py3-none-any.whl

  * pip install sdist
    * tar xzf beautifulplots-0.1.1.tar.gz
    * pip install beautifulplots-0.1.1/


Publishing to testPyPi  
    * add testPyPi to Poetry repositories
    * poetry config repositories.test-pypi https://test.pypi.org/legacy/ 
    * **poetry publish -r test-pypi** # you will need login and password


Install to a local environment  
    * login to testPyPi and navigate to your project (beautifulplots)
    * copy the install link at top e.g., pip install -i https:test.pypi.org/simple ...
    * On your comptuter activate a python virtual environment with Python 3.8 or higher
    * install with pip (as copied above)

Publishing to PyPy
    * poetry publish

# Releasing and Versioning (Chapter 7)

* git commit -m types
  * \<type> refers to the kind of change made and is usually one of:
  * feat: A new feature.
  * fix: A bug fix.
  * docs: Documentation changes.
  * style: Changes that do not affect the meaning of the code (white- space, formatting, missing semi-colons, etc).
  * refactor: A code change that neither fixes a bug nor adds a feature.
  * perf: A code change that improves performance.
  * test: Changes to the test framework.
  * build: Changes to the build process or tools.

```
