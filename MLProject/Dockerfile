FROM python:3.10-slim 

# Set direktori kerja di dalam container
WORKDIR /app

# Salin file requirements.txt dulu untuk optimasi layer Docker
COPY requirements.txt /app/requirements.txt

# Install dependensi menggunakan requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua isi dari folder MLProject (lokal) ke /app (di container)
# Ini termasuk modelling_tuning.py, conda.yaml, MLproject file, dan folder data
COPY . /app

ENTRYPOINT ["python", "modelling_tuning.py"]