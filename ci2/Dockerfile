FROM {{ base_image.image }}

COPY hailjwt/setup.py /hailjwt/
COPY hailjwt/hailjwt /hailjwt/hailjwt
RUN python3 -m pip install --no-cache-dir /hailjwt && \
  rm -rf /hailjwt

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

COPY ci2/requirements.txt .
RUN python3 -m pip install --no-cache-dir -U -r requirements.txt

# FIXME install
COPY ci2/ci /ci

EXPOSE 5000

CMD ["python3", "-c", "import ci; ci.run()"]
