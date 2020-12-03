import multiprocessing

from rx.scheduler import ThreadPoolScheduler

cpu_scheduler = ThreadPoolScheduler(multiprocessing.cpu_count())
"""Thread scheduler with as many threads as CPU cores.

Note that full parallel multi-core usage may not occur if there is more
than one task that requires continuous usage of the GIL. However, if all
CPU-bound tasks release the GIL when doing heavy computations (e.g. by
off-loading work to external libraries such as NumPy), then CPU cores
will be utilized to better effect. If this is not the case, one should
use a scheduler that utilizes multiple *processes* instead of threads.
"""

io_scheduler = ThreadPoolScheduler()
"""Thread scheduler for IO-bound tasks.

The number of workers is set to the default value given by
`concurrent.futures.ThreadPoolExecutor`, which can be found in the
official Python documentation for your Python version.
"""

schedulers = {"cpu": cpu_scheduler, "io": io_scheduler}


__all__ = [
    "cpu_scheduler",
    "io_scheduler",
    "schedulers",
]
