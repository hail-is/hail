FROM {{ base_image.image }}

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

COPY scheduler/setup.py /scheduler/
COPY scheduler/scheduler/ /scheduler/scheduler/
RUN python3 -m pip install --no-cache-dir /scheduler \
  && rm -rf /scheduler

EXPOSE 5000

CMD ["python3", "-m", "scheduler"]
