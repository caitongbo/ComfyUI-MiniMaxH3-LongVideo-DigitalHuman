from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAM = ROOT / "h3_streaming_vhs.py"


def _load_stream_module():
    spec = spec_from_file_location("_h3_lossless_final_output_test_module", STREAM)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeVHS:
    calls = []

    def combine_video(self, **kwargs):
        self.__class__.calls.append(kwargs)
        extension = "mkv" if kwargs["format"] == "video/ffv1-mkv" else "mp4"
        return {
            "ui": {
                "gifs": [{
                    "filename": f"test_00001-audio.{extension}",
                    "subfolder": "",
                    "type": "output",
                }]
            },
            "result": ((True, []),),
        }


def _run(module, pix_fmt):
    _FakeVHS.calls.clear()
    module._resolve_vhs_video_combine = lambda: _FakeVHS
    result = module._run_vhs_h264(
        frames=object(),
        audio=object(),
        filename_prefix="video/test",
        pix_fmt=pix_fmt,
        crf=19,
        save_metadata=False,
        trim_to_audio=True,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        unique_id="test",
    )
    return _FakeVHS.calls[-1], result


def test_final_output_pix_fmt_widget_adds_lossless_without_changing_default():
    module = _load_stream_module()
    schema = module._vhs_h264_inputs("video/test", True)
    choices, options = schema["pix_fmt"]
    assert choices == ["yuv420p", "yuv420p10le", "lossless_ffv1"]
    assert options["default"] == "yuv420p"
    assert schema["crf"][1]["default"] == 19
    assert schema["save_metadata"][1]["default"] is False
    assert schema["trim_to_audio"][1]["default"] is True
    assert schema["save_output"][1]["default"] is True


def test_existing_h264_output_path_is_unchanged():
    module = _load_stream_module()
    call, result = _run(module, "yuv420p")
    assert call["format"] == "video/h264-mp4"
    assert call["pix_fmt"] == "yuv420p"
    assert call["crf"] == 19
    assert call["save_metadata"] is False
    assert call["trim_to_audio"] is True
    assert call["save_output"] is True
    assert result["ui"]["images"][0]["filename"].endswith(".mp4")


def test_lossless_choice_selects_builtin_ffv1_mkv_with_16bit_rgb_without_alpha():
    module = _load_stream_module()
    call, result = _run(module, "lossless_ffv1")
    assert call["format"] == "video/ffv1-mkv"
    assert call["pix_fmt"] == "rgb48le"
    # CRF remains in the stable wrapper signature/widget layout. VHS ignores it
    # for FFV1 because the ffv1-mkv format has no CRF format widget.
    assert call["crf"] == 19
    assert result["ui"]["images"][0]["filename"].endswith(".mkv")
