import time
import psutil

def cpu_usage(cpu_usage):
    cpu_percent = (cpu_usage / 100.0)
    return cpu_percent

def ram_usage(ram_usage):
    mem_percent = (ram_usage / 100.0)
    return mem_percent

