import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True  # <- ensures config applies even in Jupyter
)