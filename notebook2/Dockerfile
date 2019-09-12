FROM {{ service_base_image.image }}

COPY web_common/setup.py /web_common/setup.py
COPY web_common/web_common /web_common/web_common/
RUN pip3 install --no-cache-dir /web_common && \
  rm -rf /web_common

COPY notebook2/setup.py /notebook/setup.py
COPY notebook2/notebook /notebook/notebook/
RUN pip3 install --no-cache-dir /notebook && \
  rm -rf /notebook

EXPOSE 5000

ENTRYPOINT ["python3", "-m", "notebook"]
