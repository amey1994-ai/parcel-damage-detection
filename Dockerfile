FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.6.0+cpu torchvision==0.21.0+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pillow

COPY app.py .
COPY best_model.pth .

EXPOSE 8000

CMD ["python", "app.py"]
