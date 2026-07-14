from app.streamer import FrameBuffer


def test_publish_and_latest():
    b = FrameBuffer()
    assert b.latest() is None
    b.publish(b"\xff\xd8jpegbytes\xff\xd9")
    assert b.latest() == b"\xff\xd8jpegbytes\xff\xd9"


def test_frames_yields_multipart_chunk():
    b = FrameBuffer()
    b.publish(b"XYZ")
    gen = b.frames()
    chunk = next(gen)
    assert b"--frame" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"XYZ" in chunk
