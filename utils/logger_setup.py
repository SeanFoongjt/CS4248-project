import logging
import os

class ImmediateFileHandler(logging.FileHandler):
    """Forces Python to write to the OS, and forces the OS to write to disk."""
    def emit(self, record):
        super().emit(record)
        self.flush() 
        try:
            os.fsync(self.stream.fileno()) 
        except OSError:
            pass # Failsafe in case the stream doesn't support raw file descriptors

def setup_logger(path="output.log"):
    # 1. Strip any existing loggers to prevent duplicate output streams
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 2. Attach ONLY the File Handler (Console is reserved for tqdm)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            ImmediateFileHandler(path, encoding='utf-8')
        ]
    )