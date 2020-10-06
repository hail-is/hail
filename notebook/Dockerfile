FROM {{ service_base_image.image }}

COPY notebook/setup.py notebook/MANIFEST.in /notebook/
COPY notebook/notebook /notebook/notebook/
RUN pip3 install --use-feature=2020-resolver --upgrade --no-cache-dir /notebook && \
  rm -rf /notebook && \
  python3 -m pip check

EXPOSE 5000

CMD ["python3", "-m", "notebook"]
