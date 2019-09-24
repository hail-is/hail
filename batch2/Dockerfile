FROM {{ service_base_image.image }}

COPY batch2/run-worker.sh /

COPY batch2/setup.py batch2/MANIFEST.in /batch2/
COPY batch2/batch2 /batch2/batch2/
RUN pip3 install --no-cache-dir /batch2 && \
  rm -rf /batch2

EXPOSE 5000

CMD ["python3", "-m", "batch2"]
