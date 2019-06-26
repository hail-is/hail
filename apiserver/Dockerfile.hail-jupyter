FROM {{ hail_base_image.image }}

COPY hail/python/setup-hailtop.py /hailtop/setup.py
COPY hail/python/hailtop /hailtop/hailtop/
RUN python3 -m pip install --no-cache-dir /hailtop \
  && rm -rf /hailtop

RUN python3 -m pip install --no-cache-dir https://github.com/hail-is/jgscm/archive/v0.1.10-hail.zip

RUN mkdir /home/jovian && \
    mkdir /home/jovian/bin && \
    groupadd jovian && \
    useradd -g jovian jovian && \
    chown -R jovian:jovian /home/jovian

USER jovian
WORKDIR /home/jovian
ENV HOME /home/jovian
ENV PATH "/home/jovian/bin:$PATH"
ENV HAIL_APISERVER_URL "http://apiserver:5000"
ENV BATCH_URL "http://batch.{{ default_ns.name }}"

COPY apiserver/jupyter_notebook_config.py.in /home/jovian/
RUN sed -e "s,@project@,{{ global.project }},g" \
  < /home/jovian/jupyter_notebook_config.py.in \
  > jupyter_notebook_config.py

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
