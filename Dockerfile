FROM python:3.11

ENV LANG C.UTF-8

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python", "app.py"]