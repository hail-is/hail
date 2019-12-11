FROM {{ service_base_image.image }}

COPY router-resolver/router_resolver /router_resolver

EXPOSE 5000

CMD ["python3", "/router_resolver/router_resolver.py"]
