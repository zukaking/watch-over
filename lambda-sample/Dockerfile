FROM public.ecr.aws/lambda/python:3.8
COPY requirements.txt ./
COPY app.py   ./
COPY haarcascade_frontalface_default.xml  ./

# Install the specified packages
RUN yum install -y mesa-libGL-devel 
RUN pip install -r requirements.txt

CMD ["app.handler"]  