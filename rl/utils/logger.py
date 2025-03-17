import logging
from datetime import datetime
from termcolor import colored


class Logger:
    
    COLORS = {
        'SUCCESS': 'green',
        'ERROR': 'red',
        'WARNING': 'yellow',
        'INFO': 'white',
        'DEBUG': 'blue'
    }

    def __init__(self, name=__name__):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.propagate = False
        
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        class ColoredFormatter(logging.Formatter):
            def format(self, record):
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                level = record.levelname
                color = Logger.COLORS.get(level, 'white')
                colored_level = colored(f"[{level}]", color, attrs=['bold'])
                colored_message = colored(record.msg, color)
                return f"{timestamp} - {colored_level} - {colored_message}"
        
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)

    def success(self, message):
        logging.addLevelName(25, 'SUCCESS')
        self.logger.log(25, message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

logger = Logger()