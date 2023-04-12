import logging
from typing import Callable

import click

LOG_COLORS = {
    logging.DEBUG: "36",  # Cyan
    logging.INFO: "32",  # Green
    logging.WARNING: "33",  # Yellow
    logging.ERROR: "31",  # Red
    logging.CRITICAL: "41",  # White on Red
}


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.color_code = LOG_COLORS.get(record.levelno, "0")
        return super().format(record)


def _set_log_level_callback(ctx: click.Context, opt: click.Option, value: str) -> None:
    """Click option callback to parse and setup logging level."""

    format = (
        "\033[%(color_code)sm%(levelname)s\t\033[0m %(name)s:%(lineno)d %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(format))
    log_level = getattr(logging, value.upper())

    logging.basicConfig(
        level=log_level,
        handlers=[handler],
    )

    opt.expose_value = False


def log_level_option() -> Callable:
    """Click option to setup the log level, does not expose (return) any object."""

    def decorator(f: Callable) -> None:
        return click.option(
            "--log-level",
            "-log",
            default="WARNING",
            help="Set log level",
            callback=_set_log_level_callback,
        )(f)

    return decorator


def interactive_option() -> Callable:
    """Click option for interactive mode, exposes a boolean variable `interactive`."""

    def decorator(f: Callable) -> None:
        return click.option("--interactive", is_flag=True, default=False)(f)

    return decorator
