FROM {{ service_base_image.image }}

COPY batch2/setup.py batch2/MANIFEST.in /batch/
COPY batch2/batch /batch/batch/
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

EXPOSE 5000

CMD ["python3", "-m", "batch.batch"]
