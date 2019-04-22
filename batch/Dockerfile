FROM {{ base_image.image }}

COPY hailjwt/setup.py /hailjwt/
COPY hailjwt/hailjwt /hailjwt/hailjwt/
RUN pip3 install --no-cache-dir /hailjwt && \
  rm -rf /hailjwt

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

EXPOSE 5000

CMD ["python3", "-c", "import batch.server; batch.server.serve()"]
