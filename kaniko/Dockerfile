FROM python:3.7-slim-stretch
RUN pip3 install jinja2
COPY jinja2_render.py /jinja2_render.py

FROM gcr.io/kaniko-project/executor:debug
COPY --from=0 / /python3.7-slim-stretch
COPY render_key /render_key
