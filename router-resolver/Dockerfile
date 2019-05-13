FROM base

RUN python3 -m pip install --no-cache-dir -U kubernetes_asyncio

COPY router-resolver /router-resolver

EXPOSE 5000

CMD ["python3", "/router-resolver/router-resolver.py"]
