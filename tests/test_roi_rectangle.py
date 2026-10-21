import pytest
import numpy as np

from roi_rectangle import RoiRectangle


def test_roi_rectangle_init():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    assert roi.x1 == 10
    assert roi.y1 == 20
    assert roi.x2 == 30
    assert roi.y2 == 40
    assert roi.width == 20
    assert roi.height == 20


def test_roi_rectangle_center():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    assert roi.center == (20, 30)


def test_roi_rectangle_move_to_center():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    roi = roi.recenter((50, 60))
    assert roi.x1 == 40
    assert roi.y1 == 50
    assert roi.x2 == 60
    assert roi.y2 == 70


def test_roi_rectangle_resize():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    roi = roi.resize(40, 60)
    assert roi.x1 == 0
    assert roi.y1 == 0
    assert roi.x2 == 40
    assert roi.y2 == 60
    assert roi.width == 40
    assert roi.height == 60


def test_roi_rectangle_to_tuple():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    assert roi.to_tuple() == (20, 40, 10, 30)  # (y1, y2, x1, x2)


def test_roi_rectangle_area():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    assert roi.area == 400


def test_roi_rectangle_shape():
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    assert roi.shape == (20, 20)


def test_roi_rectangle_slice():
    image = np.random.randint(0, 255, (100, 100))
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    sliced_image = roi.slice(image)
    assert sliced_image.shape == (20, 20)


def test_roi_rectangle_get_slices():
    image = np.random.randint(0, 255, (100, 100))
    roi = RoiRectangle(x1=10, y1=20, x2=30, y2=40)
    slices = roi.get_slices()
    sliced_image = image[slices]
    assert sliced_image.shape == (20, 20)


def test_roi_rectangle_from_tuple():
    roi = RoiRectangle.from_tuple((20, 40, 10, 30))  # (y1, y2, x1, x2)
    assert roi.x1 == 10
    assert roi.y1 == 20
    assert roi.x2 == 30
    assert roi.y2 == 40
    assert roi.width == 20
    assert roi.height == 20


def test_roi_rectangle_none_values():
    roi = RoiRectangle(x1=10, y1=20, x2=None, y2=None)
    assert roi.x1 == 10
    assert roi.y1 == 20
    assert roi.x2 is None
    assert roi.y2 is None
    assert roi.width is None
    assert roi.height is None
    assert roi.center is None
    assert roi.area is None


def test_resize_keeps_unbounded_axis_start():
    # Regression: resizing shifted x1/y1 of an unbounded (None) axis.
    roi = RoiRectangle(y1=20, y2=40, x1=10, x2=None)
    assert roi.resize(30, 10) == RoiRectangle(y1=25, y2=35, x1=10, x2=None)
    roi = RoiRectangle(y1=20, y2=None, x1=10, x2=30)
    assert roi.resize(30, 10) == RoiRectangle(y1=20, y2=None, x1=5, x2=35)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(y1=20, y2=10, x1=0, x2=10),
        dict(y1=0, y2=10, x1=20, x2=10),
    ],
)
def test_negative_size_raises(kwargs):
    with pytest.raises(ValueError):
        RoiRectangle(**kwargs)


def test_zero_size():
    roi = RoiRectangle(y1=5, y2=5, x1=7, x2=7)
    assert roi.shape == (0, 0)
    assert roi.area == 0
    assert roi.center == (7, 5)
    assert roi.slice(np.zeros((10, 10))).shape == (0, 0)


def test_center_rounds_down():
    assert RoiRectangle(y1=0, y2=3, x1=0, x2=5).center == (2, 1)


def test_partially_unbounded():
    roi = RoiRectangle(y1=2, y2=None, x1=3, x2=8)
    assert roi.width == 5
    assert roi.height is None
    assert roi.shape == (None, 5)
    assert roi.center is None
    assert roi.area is None
    assert roi.slice(np.zeros((10, 10))).shape == (8, 5)


def test_move():
    roi = RoiRectangle(y1=20, y2=40, x1=10, x2=30)
    assert roi.move(5, -3) == RoiRectangle(y1=17, y2=37, x1=15, x2=35)
    assert roi == RoiRectangle(y1=20, y2=40, x1=10, x2=30)  # unchanged
    assert RoiRectangle(1, None, 2, None).move(1, 1) == RoiRectangle(2, None, 3, None)


def test_recenter_keeps_size():
    roi = RoiRectangle(y1=0, y2=11, x1=0, x2=7).recenter((100, 50))
    assert roi.shape == (11, 7)
    assert roi.center == (100, 50)


def test_recenter_unbounded_moves_start():
    roi = RoiRectangle(y1=0, y2=None, x1=0, x2=None)
    assert roi.recenter((5, 6)) == RoiRectangle(y1=6, y2=None, x1=5, x2=None)


def test_resize_odd_size_is_exact():
    roi = RoiRectangle(y1=0, y2=10, x1=0, x2=10).resize(7, 5)
    assert roi.shape == (5, 7)
    assert roi == RoiRectangle(y1=3, y2=8, x1=2, x2=9)


def test_resize_fully_unbounded_is_noop():
    roi = RoiRectangle(y1=1, y2=None, x1=2, x2=None)
    assert roi.resize(10, 10) == roi


def test_resize_negative_raises():
    with pytest.raises(ValueError):
        RoiRectangle(y1=0, y2=10, x1=0, x2=10).resize(-4, 4)


def test_tuple_round_trip():
    roi = RoiRectangle(y1=1, y2=None, x1=3, x2=4)
    assert RoiRectangle.from_tuple(roi.to_tuple()) == roi


def test_slice_values_half_open():
    image = np.arange(100).reshape(10, 10)
    roi = RoiRectangle(y1=2, y2=4, x1=5, x2=7)
    np.testing.assert_array_equal(roi.slice(image), [[25, 26], [35, 36]])
    np.testing.assert_array_equal(image[roi.get_slices()], roi.slice(image))


def test_slice_leading_axes():
    stack = np.zeros((3, 4, 50, 60))
    roi = RoiRectangle(y1=10, y2=20, x1=5, x2=35)
    assert roi.slice(stack).shape == (3, 4, 10, 30)


def test_slice_clipped_at_image_border():
    # Extends past the bottom-right edge: numpy clips the result.
    roi = RoiRectangle(y1=90, y2=120, x1=80, x2=200)
    assert roi.slice(np.zeros((100, 100))).shape == (10, 20)
    # Entirely outside the image: empty result.
    roi = RoiRectangle(y1=200, y2=210, x1=0, x2=10)
    assert roi.slice(np.zeros((100, 100))).shape == (0, 10)


def test_slice_negative_coordinates_follow_numpy():
    # Negative coordinates are passed to numpy unchanged (count from the end).
    roi = RoiRectangle(y1=-3, y2=None, x1=0, x2=2)
    assert roi.slice(np.zeros((10, 10))).shape == (3, 2)


def test_float_coordinates():
    roi = RoiRectangle(y1=0.5, y2=2.5, x1=1.0, x2=4.0)
    assert roi.shape == (2.0, 3.0)
    assert roi.area == 6.0
    with pytest.raises(TypeError):
        roi.slice(np.zeros((10, 10)))


if __name__ == "__main__":
    pytest.main()
