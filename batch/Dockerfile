FROM {{ service_base_image.image }}

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

# https://cloud.google.com/storage/docs/gsutil/addlhelp/CRC32CandInstallingcrcmod
RUN apt-get update && \
    apt-get install -y python-pip && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir -U crcmod

EXPOSE 5000

CMD ["python3", "-m", "batch"]
