import threading
import time
import psutil
import cpuinfo

# Wrap NVML import in a try-except block to make it optional
try:
    import pynvml
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False

class SystemMonitor:
    """
    A thread-based monitor to sample system resource usage (CPU, RAM, GPU)
    at a given interval while a task is running.
    """
    def __init__(self, interval=1):
        self.interval = interval
        self.is_running = False
        self.results = []
        self._thread = None
        self.static_info = self._get_static_info()

    def _get_static_info(self):
        """Gathers static system info like CPU/GPU models once."""
        info = {}
        try:
            info['cpu_model'] = cpuinfo.get_cpu_info().get('brand_raw', 'N/A')
        except Exception:
            info['cpu_model'] = "N/A"

        gpu_models = []
        if NVIDIA_SMI_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        gpu_models.append(pynvml.nvmlDeviceGetName(handle))
                pynvml.nvmlShutdown()
            except Exception:
                pass # Fail silently
        info['gpu_models'] = ", ".join(gpu_models) if gpu_models else "N/A"
        return info

    def _monitor_loop(self):
        """The main loop for the monitoring thread."""
        while self.is_running:
            snapshot = {'timestamp': time.time()}
            # CPU and RAM
            snapshot['cpu_percent'] = psutil.cpu_percent()
            snapshot['ram_percent'] = psutil.virtual_memory().percent

            # GPU Metrics
            gpu_stats = {
                'gpu_util_percent': 0, 'gpu_mem_percent': 0, 'gpu_power_mw': 0
            }
            if NVIDIA_SMI_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        # Aggregate stats across all GPUs
                        total_util, total_mem, total_power = 0, 0, 0
                        for i in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            total_util += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            total_mem += (mem_info.used / mem_info.total * 100)
                            total_power += pynvml.nvmlDeviceGetPowerUsage(handle)
                        
                        gpu_stats['gpu_util_percent'] = total_util / device_count
                        gpu_stats['gpu_mem_percent'] = total_mem / device_count
                        gpu_stats['gpu_power_mw'] = total_power

                    pynvml.nvmlShutdown()
                except Exception:
                    pass # Fail silently if something goes wrong during sampling
            
            snapshot.update(gpu_stats)
            self.results.append(snapshot)
            time.sleep(self.interval)

    def start(self):
        """Starts the monitoring thread."""
        if self._thread is not None:
            return # Already started
        self.is_running = True
        self.results = []
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("    System monitor started.")

    def stop(self):
        """Stops the monitoring thread and returns processed results."""
        if not self.is_running:
            return {}
        self.is_running = False
        self._thread.join()
        self._thread = None
        print("    System monitor stopped.")
        return self._process_results()

    def _process_results(self):
        """Calculates summary statistics from the collected samples."""
        if not self.results:
            return {}

        num_samples = len(self.results)
        processed = {
            'avg_cpu_percent': sum(r['cpu_percent'] for r in self.results) / num_samples,
            'max_cpu_percent': max(r['cpu_percent'] for r in self.results),
            'avg_ram_percent': sum(r['ram_percent'] for r in self.results) / num_samples,
            'max_ram_percent': max(r['ram_percent'] for r in self.results),
            'avg_gpu_util_percent': sum(r['gpu_util_percent'] for r in self.results) / num_samples,
            'max_gpu_util_percent': max(r['gpu_util_percent'] for r in self.results),
            'avg_gpu_mem_percent': sum(r['gpu_mem_percent'] for r in self.results) / num_samples,
        }

        # Calculate total energy consumption
        # Energy (mWs) = Power (mW) * time (s). Interval is our time delta.
        total_energy_mWs = sum(r['gpu_power_mw'] for r in self.results) * self.interval
        # Convert milliwatt-seconds to Watt-hours (1 Wh = 3600 Ws = 3,600,000 mWs)
        total_energy_Wh = total_energy_mWs / 3_600_000 if total_energy_mWs > 0 else 0
        processed['total_gpu_energy_wh'] = total_energy_Wh

        return processed