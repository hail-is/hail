FROM {{ base_image.image }}

COPY scorecard /scorecard

EXPOSE 5000

CMD ["bash", "-c", "python3 /scorecard/scorecard.py"]
