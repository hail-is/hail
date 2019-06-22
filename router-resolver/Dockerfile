FROM {{ base_image.image }}

COPY gear/setup.py /gear/
COPY gear/gear/ /gear/gear/
RUN python3 -m pip install --no-cache-dir /gear \
  && rm -rf /gear

COPY router-resolver/router-resolver /router-resolver

EXPOSE 5000

CMD ["python3", "/router-resolver/router-resolver.py"]
