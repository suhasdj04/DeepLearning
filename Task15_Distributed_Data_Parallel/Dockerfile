FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir matplotlib pandas scikit-learn

COPY ddp_train.py /workspace/ddp_train.py

ENTRYPOINT ["python", "ddp_train.py"]
