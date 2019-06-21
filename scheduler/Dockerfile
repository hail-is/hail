FROM {{ base_image.image }}

COPY gear/setup.py /gear/
COPY gear/gear/ /gear/gear/
RUN python3 -m pip install --no-cache-dir /gear \
  && rm -rf /gear

COPY scheduler/setup.py /scheduler/
COPY scheduler/scheduler/ /scheduler/scheduler/
RUN python3 -m pip install --no-cache-dir /scheduler \
  && rm -rf /scheduler

EXPOSE 5000

CMD ["python3", "-m", "scheduler"]
