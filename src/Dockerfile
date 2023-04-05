from python:3.11

WORKDIR /src

copy requirements.txt .

run pip install -r requirements.txt
copy . .
expose 5000

ENV FLASK_APP=app.py
cmd ["python", "-m", "flask", "run" , "--host=0.0.0.0", "--port=3000"]