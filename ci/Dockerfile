FROM {{ service_base_image.image }}

# FIXME install
COPY ci/ci /ci

EXPOSE 5000

CMD ["python3", "-c", "import ci; ci.run()"]
