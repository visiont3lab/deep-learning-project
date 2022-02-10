

# Python version installed
-> 3.8.10 --> add python to the global variable

# Check python installation
python --version
pip --version

# Python virtualenv setup
pip install virtualenv
virtualenv env
source env/Scripts/activate
pip install streamlit pandas pillow numpy opencv-python
pip freeze > requirements.txt

# Run the program
streamlit run app.py


# git commands
git clone 