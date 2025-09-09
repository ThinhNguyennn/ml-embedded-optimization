import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

def get_available_cameras():
    Gst.init(None)
    available_cameras = {}
    device_monitor = Gst.DeviceMonitor.new()
    device_monitor.add_filter("Video/Source")
    devices = device_monitor.get_devices()
    for index, device in enumerate(devices):
        available_cameras[index] = device.get_display_name()
    return available_cameras

