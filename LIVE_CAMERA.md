# Live T265 display

Run from this project:

```sh
python3 scripts/live_server.py
```

Open http://127.0.0.1:4174 and click **Connect camera**. Connect the Intel RealSense T265 by USB first, and close other programs using it (including RealSense Viewer). **Disconnect** stops the tracker and releases the device. Stop the server with Ctrl+C when finished; closing the browser alone leaves the tracker running.

The server uses `/opt/anaconda3/envs/umetrack/bin/python` and imports the existing tracker from `/Users/nishiyi/Documents/FYP/Umetrack-with-YOLO-roi`. It runs native T265 stereo capture, calibration validation, YOLO-OBB, UmeTrack and the existing annotated preview. No FYP files are modified. Frames stay local and are held temporarily, not recorded.

Default YOLO model: `/Users/nishiyi/Documents/Pipeline/pipeline2-3.pt`. The UmeTrack weights are unchanged. Use `--yolo-model /absolute/path/to/model.pt` to override it. To select Pipeline 1:

```sh
python3 scripts/live_server.py --model-choice pipeline1
```

Options: `--port`, `--python`, `--fyp`, `--calibration`. If port 4174 is occupied, stop the previous static server or use `--port 4175` and open that port instead.

A plain static web server cannot start the Python tracker. Open the page through this local server. The T265 pipeline requires its stereo fisheye streams and matching calibration; a normal laptop webcam is not a drop-in replacement. The browser displays JPEG snapshots of the detector output, preserving the complete stereo image within the monitor. Display refresh is capped near 16 FPS; reported FPS comes from inference, not browser refresh.

If the connection fails, the page shows the tracker error. Check USB, the `umetrack` environment, model files and device calibration, then disconnect/reconnect. The server accepts only local, same-origin camera controls. It is intended for local demos, not public hosting.
