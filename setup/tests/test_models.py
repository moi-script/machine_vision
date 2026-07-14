from app.models import Settings, CameraSettings, PlayerIn


def test_settings_defaults_populate_from_config():
    s = Settings.defaults()
    assert s.drill.intervals["easy"] == 5.0
    assert s.drill.intervals["hard"] == 1.5
    assert len(s.court.corners) == 4
    assert s.detection.shuttleSource in ("local", "serverless", "off")


def test_camera_source_accepts_int_or_str():
    assert CameraSettings(source=0).source == 0
    assert CameraSettings(source="clip.mp4").source == "clip.mp4"


def test_player_in_requires_name():
    p = PlayerIn(name="Test Player", age=20, role="player")
    assert p.jerseyNumber >= 0
