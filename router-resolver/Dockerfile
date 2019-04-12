FROM base

RUN pip install --no-cache-dir kubernetes_asyncio

COPY router-resolver /router-resolver

EXPOSE 5000

CMD ["python3", "/router-resolver/router-resolver.py"]
