@echo on
pip uninstall -y dist/tif_model_iterator-1.0.0-py3-none-any.whl
del /Q dist\
python setup.py bdist_wheel
pip install dist/tif_model_iterator-1.0.0-py3-none-any.whl
