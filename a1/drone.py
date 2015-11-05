from dronekit import connect
import time
import requests

# Connect to the Vehicle (in this case a simulator running the same computer)
vehicle = connect('127.0.0.1:14550', await_params=True)

CANVAS_WIDTH_FEET = 18
CANVAS_HEIGHT_FEET = 9
API_BASE = 'http://drone.gotechnica.org' # TODO replace with API base
PASSWORD = '' # TODO set password

# reset the drops counter
result = requests.post(API_BASE + '/drops/reset', data={'password': PASSWORD}) 
if result.ok:
    print "Resetting drops counter to 0"

def fetch_next_coordinate():
    """
    Returns the relative coordinates in feet from the drone's position.
    Should only be called once per drop.

    -----------
    |         |
    |         |
    -----------
         x
    Where x is the drone origin relative to the canvas.
    Note: the origin (0,0) is at the top left and (width,height) is the bottom right

    drop has the following relevant attributes: id, xcoord, ycoord
    id is the id of the drop
    xcoord is the x percent on the canvas 
    ycoord is the y percent on the canvas
    (z is the altitude)
    """

    result = requests.post(API_BASE + '/drops/next', data={'password': PASSWORD}) 
    if not result.ok:
        print "error fetching request"
        return (False, False)

    drop = result.json()
    # convert percents to absolute coordinates
    x = (drop['xcoord'] / 100.0) * CANVAS_WIDTH_FEET
    y = (drop['ycoord'] / 100.0) * CANVAS_HEIGHT_FEET

    # clamp x & y inside canvas
    x = max(0, min(x, CANVAS_WIDTH_FEET))
    y = max(0, min(y, CANVAS_HEIGHT_FEET))

    # return relative coordinates
    return ((x - CANVAS_WIDTH_FEET) / 2.0, CANVAS_HEIGHT_FEET - y)

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude (in meters).
    """

    print "Basic pre-arm checks"
    # Don't let the user try to fly autopilot is booting
    if vehicle.mode.name == "INITIALISING":
        print "Waiting for vehicle to initialise"
        time.sleep(1)
    while vehicle.gps_0.fix_type < 2:
        print "Waiting for GPS...:", vehicle.gps_0.fix_type
        time.sleep(1)

    print "Arming motors"
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    vehicle.armed   = True

    while not vehicle.armed:
        print " Waiting for arming..."
        time.sleep(1)

    print "Taking off!"

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
    #  after Vehicle.commands.takeoff will execute immediately).
    while True:
        print " Altitude: ", vehicle.location.global_frame.alt
        if vehicle.location.global_frame.alt>=aTargetAltitude*0.95: #Just below target, in case of undershoot.
            print "Reached target altitude"
            break
        time.sleep(1)

# http://python.dronekit.io/examples/guided-set-speed-yaw-demo.html?highlight=land
def goto(dNorth, dEast, gotoFunction=vehicle.commands.goto):
    """
    convenience function for setting position targets in metres North and East of the current location.
    It reports the distance to the target every two seconds and completes when the target is reached.
    """
    currentLocation=vehicle.location.global_frame
    targetLocation=get_location_metres(currentLocation, dNorth, dEast)
    targetDistance=get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)

    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        remainingDistance=get_distance_metres(vehicle.location.global_frame, targetLocation)
        print "Distance to target: ", remainingDistance
        if remainingDistance<=targetDistance*0.01: #Just below target, in case of undershoot.
            print "Reached target"
            break;
        time.sleep(2)

def feet_to_meters(feet):
    METERS_PER_FOOT = 0.3048
    return feet * METERS_PER_FOOT

def meters_to_feet(meters):
    FEET_PER_METER = 3.28084
    return meters * FEET_PER_METER

def drop_paint():
    pass


DRONE_Z = feet_to_meters(7)

# flight psuedo-code
while True:
    relative_xfeet, relative_yfeet = fetch_next_coordinate()
    # no new data, sleep for a bit
    if (relative_xfeet is False):
        time.sleep(15)
        continue

    arm_and_takeoff(DRONE_Z)

    # north, east = up, right
    goto(feet_to_meters(relative_yfeet), feet_to_meters(relative_xfeet))
    drop_paint()
    goto(-feet_to_meters(relative_yfeet), -feet_to_meters(relative_xfeet))
    time.sleep(1)
    vehicle.mode = VehicleMode("LAND")
    # await continue command, paint is loaded


