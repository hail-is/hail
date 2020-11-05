FROM {{ service_base_image.image }}

COPY setup.py /tcp_proxy/
COPY tcp_proxy /tcp_proxy/tcp_proxy/
RUN pip3 install --no-cache-dir /tcp_proxy && \
  rm -rf /tcp_proxy

EXPOSE 5000
