FROM {{ service_base_image.image }}

COPY batch/setup.py batch/MANIFEST.in /batch/
COPY batch/batch /batch/batch/
RUN pip3 install --no-cache-dir /batch && \
  rm -rf /batch

RUN python3 -m pip install --no-cache-dir pyjwt==1.7.1

EXPOSE 5000
