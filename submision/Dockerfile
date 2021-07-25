FROM ubuntu:16.04
FROM python:3.7

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY data data

COPY solve.py solve.py

CMD mkdir /var/log
CMD ls -lah
CMD python solve.py --input /tmp/data --output /var/log/result
