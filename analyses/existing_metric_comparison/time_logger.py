from time import perf_counter


class log_time:
    def __init__(self, job_title, log=True):
        self.job_title = job_title
        self.log = log

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        if self.log:
            print(f"Time to complete {self.job_title}: {self.time:.3f} s")
