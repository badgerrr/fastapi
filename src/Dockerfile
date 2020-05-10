FROM python:3.7-slim-buster

WORKDIR /app

# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
#Prevents Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y python3-dev build-essential

COPY /app/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


COPY ./app /app
COPY ./data /data

EXPOSE 80

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "80", "app.main:app"]
