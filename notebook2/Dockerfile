FROM {{ service_base_image.image }}

COPY notebook2/setup.py notebook2/MANIFEST.in /notebook/
COPY notebook2/notebook /notebook/notebook/
RUN pip3 install --no-cache-dir /notebook && \
  rm -rf /notebook

EXPOSE 5000

ENTRYPOINT ["python3", "-m", "notebook"]
