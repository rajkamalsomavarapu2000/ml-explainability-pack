FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

# REQUIRED: public GitHub repo link (not zed-base)
RUN git clone https://github.com/rajkamalsomavarapu2000/ml-explainability-pack.git

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
