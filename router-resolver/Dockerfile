FROM {{ base_image.image }}

COPY hailjwt/setup.py /hailjwt/
COPY hailjwt/hailjwt /hailjwt/hailjwt/
RUN pip3 install --no-cache-dir /hailjwt && \
  rm -rf /hailjwt

COPY router-resolver/router-resolver /router-resolver

EXPOSE 5000

CMD ["python3", "/router-resolver/router-resolver.py"]
