# Detect Emotions From Posts

#### This is a machine learning algorithum to detect emotions (Angry - Happy - Sad - Neutral - Love) from humans' posts.

## How to run the code.
you have to install python > 3.7.0, you can download python for windows from here: https://www.python.org/downloads/windows/

### Create Virtual Evironment
```
pip install --upgrade virtualenv
virtualenv  envname
cd envname/
cd Scripts/
activate
cd..
cd..

pip install -r requirements.txt
```
### Run the streamlit app on localhost 
```
streamlit run app.py
```
### Run the trainning script  
```
python train.py
```
### Run the prediction script
```
python predict.py --csv path/to/file.csv
```
```
python predict.py --text any_text
```
