r"""
    Wrapper for an Xsens dot set.
"""


__all__ = ['XsensDotSet']


import asyncio
import time
from .xdc import *
from queue import Queue
import torch


_N = 10  # max 10 IMUs
_SZ = 180  # max queue size


class XsensDotSet:
    # _lock = [threading.Lock() for _ in range(_N)]   # lists are thread-safe
    _buffer = [Queue(_SZ) for _ in range(_N)]
    _is_start = False
    _pending_event = None

    @staticmethod
    def _on_device_report(message_id, message_bytes, sensor_id=-1):
        parsed = DeviceReportCharacteristic.from_bytes(message_bytes)
        print('IMU %d:' % sensor_id, parsed)

    @staticmethod
    def _on_medium_payload_report(message_id, message_bytes, sensor_id=-1):
        parsed = MediumPayloadCompleteQuaternion.from_bytes(message_bytes)
        q = XsensDotSet._buffer[sensor_id]
        if q.full():
            q.get()
        q.put(parsed)

    @staticmethod
    async def _multiple_sensor(devices: list):
        # Please use xsens dot app to synchronize the sensors first.
        # do not use asyncio.gather() and run these in parallel, it has bugs
        from functools import partial

        print('finding devices ...')
        dots = []
        for i, d in enumerate(devices):
            device = await afind_by_address(d)
            dots.append(Dot(device))
            print('\t[%d]' % i, device)

        print('connecting ...')
        for i, d in enumerate(dots):
            while True:
                try:
                    await d.aconnect(timeout=5)
                    break
                except Exception as e:
                    print('\t[%d]' % i, e)
            print('\t[%d] connected' % i)

        print('reading battery infos ...')
        for i, d in enumerate(dots):
            info = await d.abattery_read()
            print('\t[%d] %d%%' % (i, info.battery_level))

        print('starting the sensors ...')
        for i, d in enumerate(dots):
            await d.adevice_report_start_notify(partial(XsensDotSet._on_device_report, sensor_id=i))
            await d.amedium_payload_start_notify(partial(XsensDotSet._on_medium_payload_report, sensor_id=i))
            await d.aset_output_rate(60)
            await d.astart_streaming(payload_mode=3)

        print('sensor started')
        XsensDotSet._is_start = True
        while True:
            if XsensDotSet._pending_event == 0:   # close
                shutdown = False
                break
            elif XsensDotSet._pending_event == 1:   # shutdown
                shutdown = True
                break
            elif XsensDotSet._pending_event == 2:   # reset heading
                for i, d in enumerate(dots):
                    await d.areset_heading()
                print('heading is reset')
            elif XsensDotSet._pending_event == 3:   # revert heading
                for i, d in enumerate(dots):
                    await d.arevert_heading_to_default()
                print('heading is reverted to default')

            XsensDotSet._pending_event = None
            await asyncio.sleep(1)

        print('disconnecting ...')
        for i, d in enumerate(dots):
            await d.astop_streaming()
            await d.amedium_payload_stop_notify()
            await d.adevice_report_stop_notify()
            if shutdown:
                await d.apower_off()
                print('\t[%d] power off' % i)
            else:
                await d.adisconnect()
                print('\t[%d] disconnected' % i)

        XsensDotSet._is_start = False

    @staticmethod
    def _run_in_new_thread(coro):
        r"""
        Similar to `asyncio.run()`, but create a new thread.
        """
        def start_loop(_loop):
            asyncio.set_event_loop(_loop)
            _loop.run_forever()

        import threading
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=start_loop, args=(loop,))
        thread.setDaemon(True)
        thread.start()
        asyncio.run_coroutine_threadsafe(coro, loop)

    @staticmethod
    def _wait_for_pending_event():
        while XsensDotSet._pending_event is not None:
            time.sleep(0.3)

    @staticmethod
    def clear(i=-1):
        r"""
        Clear the cached measurements of the ith IMU. If i < 0, clear all IMUs.

        :param i: The index of the query sensor. If negative, clear all IMUs.
        """
        if i < 0:
            XsensDotSet._buffer = [Queue(_SZ) for _ in range(_N)]  # max 10 IMUs
        else:
            XsensDotSet._buffer[i] = Queue(_SZ)

    @staticmethod
    def is_started() -> bool:
        r"""
        Whether the sensors are connected and started.
        """
        return XsensDotSet._is_start

    @staticmethod
    def get(i: int):
        r"""
        Get the next measurements of the ith IMU. May be blocked.

        :param i: The index of the query sensor.
        :return: timestamp (seconds), quaternion (wxyz), and free acceleration (m/s^2 in the global inertial frame)
        """
        parsed = XsensDotSet._buffer[i].get(block=True)
        t = parsed.timestamp.microseconds / 1e6
        q = torch.tensor([parsed.quaternion.w, parsed.quaternion.x, parsed.quaternion.y, parsed.quaternion.z])
        a = torch.tensor([parsed.free_acceleration.x, parsed.free_acceleration.y, parsed.free_acceleration.z])
        return t, q, a

    @staticmethod
    def async_connect(devices: list):
        r"""
        Connect to the sensors and start receiving the measurements.
        Only send the connecting command but will not be blocked.

        :param devices: List of Xsens dot addresses.
        """
        print('Remember: use xsens dot app to synchronize the sensors first.')
        XsensDotSet._run_in_new_thread(XsensDotSet._multiple_sensor(devices))

    @staticmethod
    def sync_connect(devices: list):
        r"""
        Connect to the sensors and start receiving the measurements. Block until finish.

        :param devices: List of Xsens dot addresses.
        """
        XsensDotSet.async_connect(devices)
        while not XsensDotSet.is_started():
            time.sleep(1)

    @staticmethod
    def async_disconnect():
        r"""
        Stop reading and disconnect to the sensors.
        Only send the disconnecting command but will not be blocked.
        """
        XsensDotSet._pending_event = 0

    @staticmethod
    def sync_disconnect():
        r"""
        Stop reading and disconnect to the sensors. Block until finish.
        """
        XsensDotSet.async_disconnect()
        while XsensDotSet.is_started():
            time.sleep(1)

    @staticmethod
    def async_shutdown():
        r"""
        Stop reading and shutdown the sensors.
        Only send the shutdown command but will not be blocked.
        """
        XsensDotSet._pending_event = 1

    @staticmethod
    def sync_shutdown():
        r"""
        Stop reading and shutdown the sensors. Block until finish.
        """
        XsensDotSet.async_shutdown()
        while XsensDotSet.is_started():
            time.sleep(1)

    @staticmethod
    def reset_heading():
        r"""
        Reset sensor heading (yaw).
        """
        XsensDotSet._pending_event = 2
        XsensDotSet._wait_for_pending_event()

    @staticmethod
    def revert_heading_to_default():
        r"""
        Revert sensor heading to default (yaw).
        """
        XsensDotSet._pending_event = 3
        XsensDotSet._wait_for_pending_event()


# example
if __name__ == '__main__':
    # copy the following codes outside this package to run
    from articulate.utils.xsens import XsensDotSet
    from articulate.utils.bullet import RotationViewer
    from articulate.math import quaternion_to_rotation_matrix
    imus = [
        # 'D4:22:CD:00:36:80',
        # 'D4:22:CD:00:36:04',
        # 'D4:22:CD:00:32:3E',
        # 'D4:22:CD:00:35:4E',
        # 'D4:22:CD:00:36:03',
        # 'D4:22:CD:00:44:6E',
        # 'D4:22:CD:00:45:E6',
        'D4:22:CD:00:45:EC',
        'D4:22:CD:00:46:0F',
        'D4:22:CD:00:32:32',
    ]
    XsensDotSet.sync_connect(imus)
    XsensDotSet.reset_heading()
    with RotationViewer(len(imus)) as viewer:
        XsensDotSet.clear()
        for _ in range(60 * 10):  # 10s
            for i in range(len(imus)):
                t, q, a = XsensDotSet.get(i)
                viewer.update(quaternion_to_rotation_matrix(q).view(3, 3), i)
    XsensDotSet.sync_disconnect()
