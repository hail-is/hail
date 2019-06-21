FROM {{ base_image.image }}

COPY gear/setup.py /gear/
COPY gear/gear/ /gear/gear/
RUN python3 -m pip install --no-cache-dir /gear \
  && rm -rf /gear

COPY scheduler /scheduler
COPY templates /templates

EXPOSE 5000

CMD ["python3", "/scheduler/scheduler.py"]
