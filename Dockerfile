FROM python:3.8-slim-buster
WORKDIR /app
COPY requirements_server.txt requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 5000/tcp
VOLUME '/data'
COPY main.py application.conf ./
CMD [ "python3", "main.py"]