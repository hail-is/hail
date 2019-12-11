FROM {{ service_base_image.image }}

COPY scheduler/setup.py /scheduler/
COPY scheduler/scheduler/ /scheduler/scheduler/
RUN python3 -m pip install --no-cache-dir /scheduler \
  && rm -rf /scheduler

EXPOSE 5000

CMD ["python3", "-m", "scheduler"]
