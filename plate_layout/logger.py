import logging

logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)8s] --- %(message)s")

logger = logging.getLogger('plate_layout')