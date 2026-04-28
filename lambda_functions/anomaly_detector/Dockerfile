# AWS official Lambda Python image
FROM public.ecr.aws/lambda/python:3.11

# Copy requirements first (better caching)
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target "${LAMBDA_TASK_ROOT}" \
    --no-cache-dir

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Lambda entrypoint
CMD ["lambda_function.lambda_handler"]