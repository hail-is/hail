FROM {{ base_image.image }}

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

COPY batch2/setup.py batch2/MANIFEST.in /batch/
COPY batch2/batch /batch/batch/
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

EXPOSE 5000

CMD ["python3", "-m", "batch.batch"]
