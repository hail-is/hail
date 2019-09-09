FROM {{ service_base_image.image }}

COPY ci/requirements.txt .
RUN python3 -m pip install --no-cache-dir -U -r requirements.txt

# FIXME install
COPY ci/ci /ci

EXPOSE 5000

CMD ["python3", "-c", "import ci; ci.run()"]
