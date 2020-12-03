FROM {{ service_base_image.image }}

COPY notebook/setup.py notebook/MANIFEST.in /notebook/
COPY notebook/notebook /notebook/notebook/
RUN hail-pip-install /notebook && rm -rf /notebook

EXPOSE 5000

CMD ["python3", "-m", "notebook"]
