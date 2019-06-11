FROM {{ base_image.image }}

COPY scheduler /scheduler
COPY templates /templates

EXPOSE 5000

CMD ["python3", "/scheduler/scheduler.py"]
