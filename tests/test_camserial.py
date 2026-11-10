"""camserial.py against the fake camera (sim_camserial.py).   Run:  python -m unittest tests.test_camserial -v"""
import os
import time
import unittest

import camserial
from sim_camserial import FakeCam


class TestChecksum(unittest.TestCase):
    def test_known_value(self):
        self.assertEqual(camserial.fletcher16_reference(b"abcde"), 0xC8F0)    # standard Fletcher-16 test vector
        self.assertEqual(camserial.fletcher16(b"abcde"), 0xC8F0)

    def test_fast_equals_reference(self):
        for n in (0, 1, 255, 4097, 70000):
            d = os.urandom(n)
            self.assertEqual(camserial.fletcher16(d), camserial.fletcher16_reference(d), n)


class TestSerialCam(unittest.TestCase):
    def _cam(self, fake, **kw):
        cam = camserial.SerialCam(fake.port, settings={"saturation": 2, "quality": 12}, auto_reset_s=0,
                                  on_log=lambda s: None, **kw)
        self.addCleanup(cam.close)
        self.addCleanup(fake.stop)
        return cam

    def test_stream_settings_and_commands(self):
        fake = FakeCam(fps=30).start()
        cam = self._cam(fake)
        f = cam.wait_frame(timeout=3)
        self.assertIsNotNone(f, cam.status)
        img = f.decode()
        self.assertEqual(img.shape, (480, 640, 3))                              # hardware-JPEG mode: full frame
        self.assertEqual(camserial.middle_third(img).shape, (160, 640, 3))      # band cropped on the PC
        self.assertEqual(camserial.middle_third(img[160:320]).shape, (160, 640, 3))   # a band stays a band
        self.assertEqual(fake.settings, {"saturation": 2, "quality": 12})      # applied on connect
        self.assertEqual(cam.command("ping"), "ok pong")
        self.assertTrue(cam.set("brightness", -1).startswith("ok"))
        self.assertEqual(fake.settings["brightness"], -1)
        f2 = cam.wait_frame(timeout=2)
        self.assertGreater(f2.ts_cam, f.ts_cam)                                 # camera timestamps move on

    def test_resync_after_junk_and_corruption(self):
        fake = FakeCam(fps=60, junk=0.3).start()
        cam = self._cam(fake)
        t_end, n = time.time() + 3, 0
        while time.time() < t_end and n < 40:
            f = cam.wait_frame(timeout=1)
            if f is not None:
                self.assertIsNotNone(f.decode())                                # every delivered frame is intact
                n += 1
        self.assertGreaterEqual(n, 40)
        self.assertGreater(cam.stats["bad"], 0)                                 # corruption was seen and skipped

    def test_reconfigure_after_reboot(self):
        fake = FakeCam(fps=20).start()
        cam = self._cam(fake)
        self.assertIsNotNone(cam.wait_frame(timeout=3))
        fake.settings.clear()
        fake.streaming = False
        fake._send(b"L", b"ready (rebooted)")                                  # what the firmware says at boot
        self.assertIsNotNone(cam.wait_frame(timeout=3))                         # stream restarted by SerialCam
        self.assertEqual(fake.settings.get("saturation"), 2)                    # settings re-applied


if __name__ == "__main__":
    unittest.main()
