"""Pick puncta by hand on a channel's original image, then warp them.

In a session with a saved registration, "Open Original Image" loads one raw
channel and its raw-space puncta as temporary layers, placed over the aligned
data by the rigid part of the registration. Their data stays in raw voxel
coordinates, so the puncta editor and the Fishing Hook work on them as on any
raw layer and every pick is a raw-space pick. "Warp Picks into Aligned Space"
takes the raw puncta through the saved registration onto the channel's aligned
puncta layer (``stages.transform_channel``) and closes the temporary layers.
"""
import logging
import threading
from collections import deque

import napari
from magicgui.widgets import CheckBox, ComboBox, Container, Label, PushButton
from qtpy.QtWidgets import QApplication, QLabel

from ... import constants
from ...core import io, puncta, stages
from .. import events, popups, viewer_helpers
from ..decorators import error_handler
from ._shared import make_divider as _make_divider

logger = logging.getLogger(__name__)

_DESCRIPTION = (
    "Load one channel's original image and its puncta over the aligned data, pick or correct puncta "
    "by hand with the Puncta Editor (the Fishing Hook works too), then warp them into aligned space "
    "with the session's saved registration. The channel's aligned layers are hidden while you pick. "
    "For R2 only the rigid shift is shown: the warp moves the aligned image a little further.")


class RawPickingWidget(Container):
    """The "Pick on Original Image" page of Puncta Picking."""

    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self._viewer = viewer
        self._active = None           # round, channel, temporary layers, hidden layer names
        self._init_widgets()
        self._init_layout()
        self.open_btn.changed.connect(self._on_open)
        self.warp_btn.changed.connect(self._on_warp)
        self.close_btn.changed.connect(self._on_close)
        self._update_buttons()

    def _init_widgets(self):
        self._header = Label(value="Pick on Original Image")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(value="<i>Pick puncta on the original image, then warp them into aligned space.</i>")
        self._info.native.setObjectName("widgetInfo")
        self.channel = ComboBox(label="Channel:", choices=self._channel_choices,
                                tooltip="The channel to pick on: one with aligned puncta in this session.")
        self.show_mask = CheckBox(text="Show the round's nuclei mask",
                                  tooltip="Also load the round's own nuclei mask (about 1.2 GB more memory "
                                          "for a typical stack), placed like the original image.")
        self.open_btn = PushButton(text="Open Original Image")
        self.warp_btn = PushButton(text="Warp Picks into Aligned Space")
        self.close_btn = PushButton(text="Close Without Warping")
        self.close_btn.tooltip = ("Remove the original image. Picks stay saved in the channel's raw puncta; "
                                  "its aligned puncta are then out of date until warped.")
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("margin: 4px 2px; font-size: 12px;")
        self._unavailable = QLabel("")
        self._unavailable.setWordWrap(True)
        self._unavailable.setStyleSheet("color: #f9e2af; margin: 4px 2px; font-size: 12px;")
        self._unavailable.setVisible(False)

    def _init_layout(self):
        desc = QLabel(_DESCRIPTION)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: white; margin: 4px 2px;")
        form = Container(labels=True)
        form.extend([self.channel])
        layout = self.native.layout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self._header.native, self._info.native, _make_divider(), desc, self._unavailable, form.native,
                       self.show_mask.native, self.open_btn.native, self.warp_btn.native,
                       self.close_btn.native, self._status):
            layout.addWidget(widget)
        layout.addStretch(1)

    # --- choices and state -------------------------------------------------------------

    def _channel_choices(self, widget=None):
        """Channels with an aligned puncta layer in the viewer ('R2 - FITC')."""
        keys = []
        for layer in self._viewer.layers:
            if (isinstance(layer, napari.layers.Points) and layer.name.endswith(constants.PUNCTA_SUFFIX)
                    and layer.name.startswith((constants.ALIGNED_PREFIX, constants.WARPED_PREFIX))):
                key = puncta.channel_key(layer.name)
                if key and key not in keys:
                    keys.append(key)
        return keys

    def reset_choices(self):
        self.channel.reset_choices()
        self._update_buttons()

    def _update_buttons(self):
        open_ = self._active is not None
        ok, reason = (True, None)
        if not open_ and self._channel_choices():
            ok, reason = stages.can_rewarp()
        self._unavailable.setText(reason or "")
        self._unavailable.setVisible(not ok)
        self.open_btn.enabled = not open_ and ok
        self.channel.enabled = not open_
        self.show_mask.enabled = not open_
        self.warp_btn.enabled = open_
        self.close_btn.enabled = open_

    def _aligned_layer(self, key, kind):
        """The aligned or warped image (``kind='image'``) or puncta layer of ``key``."""
        suffix = constants.PUNCTA_SUFFIX if kind == "puncta" else ""
        for prefix in (constants.WARPED_PREFIX, constants.ALIGNED_PREFIX):
            name = f"{prefix} {key}{suffix}"
            if name in self._viewer.layers:
                return self._viewer.layers[name]
        return None

    def _run(self, title, work):
        """Run ``work(progress)`` off the main thread under a progress dialog."""
        result, error, progress = [None], [None], deque()

        def run():
            try:
                result[0] = work(lambda p, t: progress.append((p, t)))
            except Exception as exc:
                error[0] = exc
        with popups.ProgressDialog(self._viewer.window._qt_window, title) as dialog:
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            while thread.is_alive():
                thread.join(timeout=0.05)
                while progress:
                    dialog.update_progress(*progress.popleft())
                QApplication.processEvents()
            while progress:
                dialog.update_progress(*progress.popleft())
        if error[0]:
            raise error[0]
        return result[0]

    # --- actions -----------------------------------------------------------------------

    @error_handler("Open Original Image Failed")
    def _on_open(self, _=None):
        key = self.channel.value
        if not key:
            self._say("Choose a channel with aligned puncta first.")
            return
        ok, reason = stages.can_rewarp()
        if not ok:
            popups.show_warning_popup(self._viewer.window._qt_window, "Cannot Pick on the Original Image", reason)
            return
        round_id, channel = key.split(" - ", 1)
        raw_name = stages.raw_puncta_name(round_id, channel)
        if key in self._viewer.layers or raw_name in self._viewer.layers:
            self._say(f"{key} is already open on its original image.")
            return
        image_layer = self._aligned_layer(key, "image")
        if image_layer is None:
            self._say(f"There is no aligned image of {key} to place the original over.")
            return
        image_path = popups.resolve_raw_path(self._viewer.window._qt_window, round_id)
        if image_path is None:
            self._say(f"The original {round_id} image was not found; nothing was opened.")
            return
        scale, translate = stages.raw_overlay(round_id, image_layer.scale)
        mask_path = stages.round_mask_path(round_id) if self.show_mask.value else None

        def load(progress):
            progress(10, f"Reading the original {key} image...")
            image = io.read_channel(image_path, channel)
            mask = None
            if mask_path is not None:
                progress(60, f"Reading the {round_id} nuclei mask...")
                mask = io.read_label_tif(mask_path)
            progress(100, "Done.")
            return image, mask
        image, mask = self._run(f"Opening the original {key} image...", load)

        hidden = [layer.name for layer in self._viewer.layers if layer.visible
                  and layer.name.startswith((constants.ALIGNED_PREFIX, constants.WARPED_PREFIX))
                  and puncta.channel_key(layer.name) == key]
        for name in hidden:
            self._viewer.layers[name].visible = False
        added = []
        if mask is not None:
            added.append(self._viewer.add_labels(mask, name=f"{round_id} nuclei (original image)",
                                                 scale=scale, translate=translate, opacity=0.3,
                                                 metadata={events.OVERLAY_KEY: True}))
        added.append(self._viewer.add_image(image, name=key, scale=scale, translate=translate,
                                            colormap=constants.CHANNEL_COLORS.get(channel.upper(), "gray"),
                                            blending="additive", metadata={events.OVERLAY_KEY: True}))
        csv = stages.raw_puncta_csv(round_id, channel)
        viewer_helpers._load_points_layer(self._viewer, raw_name, csv, scale, {"subtype": "puncta_csv"},
                                          translate)
        points = self._viewer.layers[raw_name]
        points.visible = True
        added.append(points)
        self._viewer.layers.selection.active = points
        self._active = {"round": round_id, "channel": channel, "key": key, "layers": added,
                        "hidden": hidden, "image_layer": image_layer}
        self._update_buttons()
        self._say(f"Picking on the original {key} image ({len(points.data)} puncta). Add or correct puncta "
                  f"with the Puncta Editor, then warp them into aligned space.")

    @error_handler("Warp Picks Failed")
    def _on_warp(self, _=None):
        active = self._active
        if active is None:
            return
        round_id, channel, key = active["round"], active["channel"], active["key"]
        points = self._viewer.layers[stages.raw_puncta_name(round_id, channel)]
        events.save_puncta_layer(points)               # every pick on disk before warping
        stages.record_raw_picks(round_id, channel)
        from .mask_editor_widget import flush_pending_mask_saves
        flush_pending_mask_saves()
        consensus = viewer_helpers.find_consensus_layer(self._viewer)
        consensus_data = consensus.data if consensus is not None else None

        def warp(progress):
            progress(10, f"Warping {key} into aligned space...")
            return stages.transform_channel(round_id, channel, keep_aligned_manual=True,
                                            consensus_mask=consensus_data)
        aligned_csv = self._run(f"Warping {key}...", warp)
        self._close()
        if aligned_csv is not None:
            image_layer = active["image_layer"]
            name = f"{image_layer.name}{constants.PUNCTA_SUFFIX}"
            layer = viewer_helpers.show_puncta_csv(self._viewer, name, aligned_csv, image_layer)
            layer.visible = True
            n = len(layer.data)
        else:
            n = 0
        note = stages.report_note()
        self._say(f"Warped {key}: {n} puncta in aligned space." + (f" {note}" if note else ""))
        from .colocalization_widget import refresh_report_note
        refresh_report_note()

    @error_handler("Close Original Image Failed")
    def _on_close(self, _=None):
        active = self._active
        if active is None:
            return
        raw_name = stages.raw_puncta_name(active["round"], active["channel"])
        if raw_name in self._viewer.layers:
            events.save_puncta_layer(self._viewer.layers[raw_name])     # the last pick too
        self._close()
        item = next((i for i in stages.status() if i.round == active["round"] and i.channel == active["channel"]),
                    None)
        stale = item is not None and item.state == stages.STALE
        self._say(f"Closed the original {active['key']} image."
                  + (" Your picks are saved in its raw puncta; its aligned puncta are out of date until "
                     "you warp them." if stale else ""))

    def _close(self):
        """Remove the temporary layers, keeping their files, and show the
        channel's aligned layers again."""
        active, self._active = self._active, None
        # Point the Puncta Editor at the channel's aligned puncta first: when the
        # layer it is on goes, it would move to another layer and show that one.
        from .puncta_editor_widget import _puncta_editor_widget
        aligned = self._aligned_layer(active["key"], "puncta")
        if aligned is not None:
            try:
                _puncta_editor_widget.points_layer.value = aligned
            except (ValueError, AttributeError):
                pass
        events._set_programmatic_removal(True)
        try:
            for layer in reversed(active["layers"]):
                if layer in self._viewer.layers:
                    layer._locked = False
                    layer.visible = False
                    if isinstance(layer, napari.layers.Points):
                        viewer_helpers.clear_points_index_cache(layer)
                    events.remove_layer_keeping_file(self._viewer, layer)
        finally:
            events._set_programmatic_removal(False)
        for name in active["hidden"]:
            if name in self._viewer.layers:
                self._viewer.layers[name].visible = True
        self._update_buttons()

    def _say(self, text):
        self._status.setText(text)
        self._viewer.status = text
