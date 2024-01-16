import logging
from nanodiarization.constants import LOG_LEVEL

module_name = __name__.split('.')[0]
logger = logging.getLogger(module_name)
logger.setLevel(LOG_LEVEL)