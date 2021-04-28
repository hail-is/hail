FROM python:3.7-slim-stretch
RUN pip3 install jinja2

FROM gcr.io/kaniko-project/executor:debug

COPY --from=0 /bin/ /python3.7-slim-stretch/bin
# boot is empty
# cannot copy dev in kaniko
# etc is too big in kaniko, ld is necessary for python to dynamically link
COPY --from=0 /etc/ld.so.cache /python3.7-slim-stretch/etc/ld.so.cache
COPY --from=0 /etc/ld.so.conf /python3.7-slim-stretch/etc/ld.so.conf
# home is empty
COPY --from=0 /lib/ /python3.7-slim-stretch/lib
COPY --from=0 /lib64/ /python3.7-slim-stretch/lib64
# media is empty
# mnt is empty
# opt is empty
# cannot copy proc in kaniko
COPY --from=0 /root/ /python3.7-slim-stretch/root
COPY --from=0 /run/ /python3.7-slim-stretch/run
COPY --from=0 /sbin/ /python3.7-slim-stretch/sbin
# srv is empty
# cannot copy sys in kaniko
# ignore tmp
COPY --from=0 /usr/ /python3.7-slim-stretch/usr
COPY --from=0 /var/ /python3.7-slim-stretch/var

COPY jinja2_render.py /python3.7-slim-stretch/jinja2_render.py
COPY kaniko/convert-google-application-credentials-to-kaniko-auth-config /convert-google-application-credentials-to-kaniko-auth-config
