
FROM python:3.12.13-slim AS databuilder
ENV PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir numpy==2.5.1
COPY make_data.py /make_data.py
COPY mnist-raw /build/mnist-raw
RUN python /make_data.py /build/mnist-raw

FROM python:3.12.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MNISTBOARD_PYTHON_VERSION=3.12.13 \
    MNISTBOARD_CUDA_VERSION=12.4 \
    MNISTBOARD_TORCH_VERSION=2.5.1+cu124 \
    MNISTBOARD_TORCHVISION_VERSION=0.20.1+cu124

COPY cpu_env.sh /usr/local/bin/cpu_env.sh
RUN chmod +x /usr/local/bin/cpu_env.sh && \
    ln -s /usr/local/bin/cpu_env.sh /etc/profile.d/cpu_env.sh && \
    echo '. /usr/local/bin/cpu_env.sh' >> /etc/bash.bashrc

ENV BASH_ENV=/usr/local/bin/cpu_env.sh \
    MNISTBOARD_AGENTS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MNISTBOARD_GPU_MEMORY_LIMIT_MIB=1000 \
    MNISTBOARD_GPU_ALLOCATOR_MIB=400

RUN pip install --no-cache-dir numpy==2.5.1 && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1+cu124 torchvision==0.20.1+cu124 && \
    pip install --no-cache-dir \
        scipy==1.18.0 scikit-learn==1.9.0 pillow==11.3.0 tqdm==4.70.0 && \
    python -c "import platform, numpy, scipy, sklearn, PIL, torch, torchvision, tqdm; \
assert platform.python_version() == '3.12.13'; \
assert numpy.__version__ == '2.5.1'; \
assert scipy.__version__ == '1.18.0'; \
assert sklearn.__version__ == '1.9.0'; \
assert PIL.__version__ == '11.3.0'; \
assert torch.__version__ == '2.5.1+cu124'; \
assert torchvision.__version__ == '0.20.1+cu124'; \
assert tqdm.__version__ == '4.70.0'; \
assert torch.version.cuda == '12.4'"

COPY --from=databuilder /data /data

WORKDIR /app
RUN mkdir -p /app/submission
COPY submission_template.py /app/submission_template.py
COPY model_spec.py /app/model_spec.py
COPY check_submission.py /app/check_submission.py
COPY check_test.py /app/check_test.py
COPY score.sh /app/score.sh
RUN chmod +x /app/score.sh
