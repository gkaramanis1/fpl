FROM docker.io/python:3.9

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 8000
ENTRYPOINT python3 app.py
# CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]