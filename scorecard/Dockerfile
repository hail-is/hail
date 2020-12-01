FROM {{ service_base_image.image }}

COPY scorecard/setup.py scorecard/MANIFEST.in /scorecard/
COPY scorecard/scorecard /scorecard/scorecard/
RUN hail-pip-install /scorecard && rm -rf /scorecard

EXPOSE 5000

CMD ["python3", "-m", "scorecard"]
