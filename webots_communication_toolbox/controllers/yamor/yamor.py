"""communicator_test controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, LED, DistanceSensor
from controller import Robot
from math import pi,sin
import numpy as np
import pickle

robot = Robot()
timestep = int(robot.getBasicTimeStep())

receiver = robot.getReceiver('receiver')
receiver.enable(timestep)





F = 1.0;            # frequency
A = 0.9;            # amplitude

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
t = 0.0
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  led = robot.getLED('ledname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

def square_8_modules():
    lookup = [ 1, 1, 0, 0, 1, 1, 0, 0 ];
    id = int(robot.getName()[-1])

    #wb_motor_set_position(motor, -1.5708 * LOOKUP[(id + (int) t) % 8]);
    robot.getMotor("motor").setPosition(-1.5708*lookup[(id)*t % 8])

def worm_8_modules():
    id = int(robot.getName()[-1])-1
    shift = id * (2.0 * pi / 8);  # phase shift for this module
    phase = 2.0 * pi * F * t;
    robot.getMotor("motor").setPosition(A * sin(phase + shift))

def rl_8_modules(actions):
    id = int(robot.getName()[-1])-1
    shift = actions[id];  # phase shift for this module
    if(type(shift) == np.ndarray):
        shift = shift[0]
    phase = 2.0 * pi * F * t;

    factor = 1.0
    if factor*shift > 1.5:
        shift = 1.5*factor;
    if factor*shift < -1.5:
        shift = -1.5*factor;
    #robot.getMotor("motor").setPosition(factor*shift)
    robot.getMotor("motor").setPosition(A * sin(phase + shift))




# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    actions = np.random.randn(8,1)
    while receiver.getQueueLength() > 0:
         actions = pickle.loads(receiver.getData())
         # do something with the map
         receiver.nextPacket()
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.
    rl_8_modules(actions)
    # Enter here functions to send actuator commands, like:
    #  led.set(1)
    t += timestep / 1000.0;

# Enter here exit cleanup code.
