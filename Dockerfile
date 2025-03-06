# # Use a lightweight Python base image
# FROM python:3.9-slim

# # Set the working directory inside the container
# WORKDIR /app

# # Install necessary system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# # Install only required Python dependencies
# RUN pip install --no-cache-dir autogluon.timeseries~=1.2.0

# # Copy the Python script into the container
# COPY chronos_example.py /app/chronos_example.py

# # Set the command to run the script
# CMD ["python", "/app/chronos_example.py"]

# Use a Python base image (version 3.10 or higher)
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies (including Git & SSH)
RUN apt update && apt install -y git openssh-client

# Copy the contents of the current directory (your code) into the container
COPY . .

# Install any dependencies if you have a requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your model
CMD ["bash"]