/* Coalesce refresh pressure while guaranteeing the newest request is served. */

export function coalesceLatest(task, shouldRun = () => true) {
  let requested = 0;
  let completed = 0;
  let running = null;

  return function request() {
    requested += 1;
    if (!shouldRun()) return Promise.resolve();
    if (!running) {
      running = (async () => {
        while (completed < requested && shouldRun()) {
          const target = requested;
          await task();
          completed = target;
        }
      })().finally(() => { running = null; });
    }
    return running;
  };
}
