# scan_wall.py - zigzag (lawnmower) inspection scan of the defect wall
#   run alone:      python scripts/scan_wall.py
#   or imported:    from scan_wall import run_scan
from flight_utils import Drone

SCAN_X = 1.5                   # flight line: 1.0 m in front of the wall (wall face at x = 2.5)
Y_LEFT, Y_RIGHT = 2.6, -2.6    # wall edges as seen from the drone
ROWS_Z = [0.55, 1.50, 2.45]    # one sweep per row of panels, bottom to top
SCAN_SPEED = 0.35              # m/s along the wall (slow = sharper camera frames)
TRAVEL_SPEED = 0.5             # m/s for getting into position


def run_scan(d):
    d.move(0, 0, ROWS_Z[0], speed=TRAVEL_SPEED); d.hold(1)
    print("Took off")
    d.move(SCAN_X, Y_LEFT, ROWS_Z[0], speed=TRAVEL_SPEED); d.hold(1)
    print("At scan start (left end of bottom row)")

    going_right = True
    for i, z in enumerate(ROWS_Z, start=1):
        if i > 1:
            d.move(SCAN_X, d.pos[1], z, speed=TRAVEL_SPEED); d.hold(1)   # climb to next row
        end_y = Y_RIGHT if going_right else Y_LEFT
        print(f"Row {i}: z={z:.2f} m, sweeping to y={end_y:+.1f}")
        d.move(SCAN_X, end_y, z, speed=SCAN_SPEED); d.hold(1)
        going_right = not going_right

    print("Scan complete - returning home")
    d.move(0, 0, 1.0, speed=TRAVEL_SPEED); d.hold(1)
    d.land()


if __name__ == "__main__":
    with Drone(name="scan") as d:
        run_scan(d)
    print("Mission finished.")