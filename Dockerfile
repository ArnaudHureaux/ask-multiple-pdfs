FROM python:3.10-slim
EXPOSE 8080
RUN apt-get update && apt-get install -y git
COPY requirements.txt app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]