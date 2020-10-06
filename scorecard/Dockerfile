FROM {{ service_base_image.image }}

COPY scorecard/setup.py scorecard/MANIFEST.in /scorecard/
COPY scorecard/scorecard /scorecard/scorecard/
RUN python3 -m pip install --use-feature=2020-resolver --upgrade --no-cache-dir /scorecard && \
  rm -rf /scorecard && \
  python3 -m pip check

EXPOSE 5000

CMD ["python3", "-m", "scorecard"]
