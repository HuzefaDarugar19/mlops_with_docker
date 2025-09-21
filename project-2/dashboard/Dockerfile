FROM python:3.9-slim

WORKDIR /app

<<<<<<< HEAD
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
=======
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt
>>>>>>> 74564bc6d69fecb53cef660e16459815ac75ba56

COPY . .

CMD ["python", "dash_app.py"]
