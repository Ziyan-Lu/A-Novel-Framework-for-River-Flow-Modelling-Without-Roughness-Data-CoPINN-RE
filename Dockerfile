FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["python", "main.py"]