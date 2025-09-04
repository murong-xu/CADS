import time

try:
    import torch
except Exception:
    torch = None

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _bytes2mib(x):  # bytes -> MiB
    return x / (1024**2)

def gpu_info_str():
    if torch is None or not torch.cuda.is_available():
        return "GPU: N/A (CPU mode)"
    i = torch.cuda.current_device()
    name = torch.cuda.get_device_name(i)
    total = torch.cuda.get_device_properties(i).total_memory
    free, tot = torch.cuda.mem_get_info()
    return (f"GPU {i} {name} | total={_bytes2mib(tot):.1f} MiB "
            f"| free={_bytes2mib(free):.1f} MiB | used={_bytes2mib(tot-free):.1f} MiB")

def log_gpu_state(tag, logfile=None):
    msg = f"[{_now()}] [{tag}] " + gpu_info_str()
    print(msg)
    if logfile:
        with open(logfile, "a") as f:
            f.write(msg + "\n")

class VRAMMonitor:
    """Capture per-block CUDA peak memory."""
    def __init__(self, tag, logfile=None, sync=True):
        self.tag = tag
        self.logfile = logfile
        self.sync = sync
    def __enter__(self):
        if torch and torch.cuda.is_available():
            if self.sync: torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._start_free, self._start_total = torch.cuda.mem_get_info()
            log_gpu_state(self.tag + " [start]", self.logfile)
        else:
            print(f"[{_now()}] [{self.tag}] CPU mode (no CUDA)")
        return self
    def __exit__(self, exc_type, exc, tb):
        if torch and torch.cuda.is_available():
            if self.sync: torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_resvd = torch.cuda.max_memory_reserved()
            cur_free, cur_total = torch.cuda.mem_get_info()
            msg = (f"[{_now()}] [{self.tag} [end]] "
                   f"peak_alloc={_bytes2mib(peak_alloc):.1f} MiB, "
                   f"peak_reserved={_bytes2mib(peak_resvd):.1f} MiB, "
                   f"free={_bytes2mib(cur_free):.1f} MiB / total={_bytes2mib(cur_total):.1f} MiB")
            print(msg)
            if self.logfile:
                with open(self.logfile, "a") as f:
                    f.write(msg + "\n")
