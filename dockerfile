# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Command to run Gunicorn serving your Flask app
# Replace 'app:app' with your module and Flask app variable
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]