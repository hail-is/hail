FROM {{ base_image.image }}

COPY gear/setup.py /gear/
COPY gear/gear/ /gear/gear/
RUN python3 -m pip install --no-cache-dir /gear \
  && rm -rf /gear

COPY batch_client/setup.py /batch_client/
COPY batch_client/batch_client /batch_client/batch_client
RUN pip3 install --no-cache-dir /batch_client && \
  rm -rf /batch_client

COPY builder/setup.py /builder/
COPY builder/builder /builder/builder
RUN python3 -m pip install --no-cache-dir /builder && \
  rm -rf /builder

COPY ci/requirements.txt .
RUN python3 -m pip install --no-cache-dir -U -r requirements.txt

# FIXME install
COPY ci/ci /ci

EXPOSE 5000

CMD ["python3", "-c", "import ci; ci.run()"]
