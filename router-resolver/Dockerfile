FROM {{ base_image.image }}

COPY router-resolver /router-resolver

EXPOSE 5000

CMD ["python3", "/router-resolver/router-resolver.py"]
