FROM {{ service_base_image.image }}

COPY setup.py MANIFEST.in /scorecard/
COPY scorecard /scorecard/scorecard/
RUN pip3 install --no-cache-dir /scorecard && \
  rm -rf /scorecard

EXPOSE 5000

CMD ["python3", "-m", "scorecard"]
