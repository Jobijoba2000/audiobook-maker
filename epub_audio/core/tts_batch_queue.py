"""Schedule bounded synthesis windows; return futures in reading order."""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event


def group_jobs_by_length(jobs, batch_size):
    """Sort longest first to fill large groups, then prioritize early book positions."""
    indexed = sorted(enumerate(jobs), key=lambda entry: (-len(entry[1][0]), entry[0]))
    batches = [sorted(indexed[first:first + batch_size])
               for first in range(0, len(indexed), batch_size)]
    # Rows retain their relative book order; the earliest needed group runs first.
    return sorted(batches, key=lambda batch: batch[0][0])


class BatchedSynthesisPool:
    def __init__(self, engine, *, lookahead_batches=8, continuous=None):
        if type(lookahead_batches) is not int or not 1 <= lookahead_batches <= 8:
            raise ValueError('Fenêtre de batching invalide.')
        self.batch_size = engine.batch_size
        self.window_size = self.batch_size * lookahead_batches
        self.engine = engine
        self.continuous = getattr(engine, 'continuous_batching', False) if continuous is None else continuous
        self.stopping = Event()
        self.pool = ThreadPoolExecutor(max_workers=1, initializer=getattr(engine, 'initialize_worker', None))

    def submit_window(self, jobs, *, on_complete=None):
        """Reuse free slots or group nearby lengths; map futures to source slots."""
        jobs = list(jobs)
        if len(jobs) > self.window_size:
            raise ValueError('La fenêtre contient trop de passages.')
        if self.continuous:
            return self._submit(jobs, self._run_continuous, on_complete)
        ordered = [None] * len(jobs)
        try:
            for batch in group_jobs_by_length(jobs, self.batch_size):
                futures = self.submit_batch([job for _, job in batch], on_complete=on_complete)
                for (index, _), future in zip(batch, futures):
                    ordered[index] = future
        except BaseException:
            for future in ordered:
                if future is not None:
                    future.cancel()
            raise
        return ordered

    def submit_batch(self, jobs, *, on_complete=None):
        return self._submit(jobs, self._run, on_complete)

    def _submit(self, jobs, run, on_complete):
        futures = [Future() for _ in jobs]
        driver = self.pool.submit(run, list(zip(jobs, futures)), on_complete)

        def propagate_cancellation(done):
            if done.cancelled():
                for future in futures:
                    future.cancel()
            elif done.exception() is not None:
                for future in futures:
                    if not future.done():
                        future.set_exception(done.exception())
        driver.add_done_callback(propagate_cancellation)
        return futures

    def _run(self, jobs, on_complete=None):
        active = [(job, future) for job, future in jobs if future.set_running_or_notify_cancel()]
        if not active:
            return
        reported = set()
        def notify(index, state):
            if not 0 <= index < len(active):
                raise ValueError('Indice de progression vocale invalide.')
            job = active[index][0]
            if state == 'ready':
                if index not in reported and on_complete is not None:
                    on_complete([job])
                reported.add(index)
            elif on_complete is not None:
                callback = getattr(on_complete, state, None)
                if callback is not None:
                    callback([job])

        try:
            start = getattr(on_complete, 'start', None)
            if start is not None:
                start([job for job, _ in active])
            notifications = ({'on_progress': notify}
                             if getattr(self.engine, 'individual_progress', False) is True else {})
            results = self.engine.synthesize_batch([job[0] for job, _ in active],
                                                   [job[1] for job, _ in active], **notifications)
            if len(results) != len(active):
                raise RuntimeError('Le groupe vocal ne contient pas tous les passages attendus.')
            remaining = [job for index, (job, _) in enumerate(active) if index not in reported]
            if remaining and on_complete is not None:
                # Engines without individual notifications only report after
                # the group really finishes; do not simulate early events.
                on_complete(remaining)
        except BaseException as exc:
            for _, future in active:
                future.set_exception(exc)
        else:
            for (_, future), result in zip(active, results):
                future.set_result(result)

    def _run_continuous(self, jobs, on_complete):
        active = [(job, future) for job, future in jobs if future.set_running_or_notify_cancel()]
        if not active:
            return
        ready = []
        completed = set()
        def flush():
            if not ready:
                return
            if on_complete is not None:
                on_complete([active[index][0] for index, _ in ready])
            for index, path in ready:
                active[index][1].set_result(path)
            ready.clear()
        def receive(index, path):
            if index in completed:
                raise ValueError('Un passage continu a été livré deux fois.')
            completed.add(index)
            ready.append((index, path))
            # Keep display increments at the configured slot count, even though
            # individual slots now finish and refill independently.
            if len(ready) >= self.batch_size:
                flush()
        try:
            results = self.engine.synthesize_continuous([job[0] for job, _ in active],
                [job[1] for job, _ in active], on_ready=receive, should_stop=self.stopping.is_set)
            if len(results) != len(active):
                raise RuntimeError('La synthèse continue n’a pas renvoyé tous les passages.')
            for index, path in enumerate(results):
                if index not in completed:
                    receive(index, path)
            flush()
        except BaseException as exc:
            for _, future in active:
                if not future.done():
                    future.set_exception(exc)

    def shutdown(self, wait=True, cancel_futures=False):
        if cancel_futures:
            self.stopping.set()
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)
