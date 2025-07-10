
import sys
import serial
import os
import time
import Hobot.GPIO as GPIO
import multiprocessing

class drone_result:
    def __init__(self, success:bool, message:str):
        self.success = success
        self.message = message

class drone_command_type(int) :
    TAKE_OFF = 0x00
    MOVE = 0x01
    LAND = 0x02

class drone_task_state(int) :
    EMPTY = 0x00
    WAITING = 0x01
    DOING = 0x02
    DONE = 0x03

class drone_direction(int) :
    FORWARD = 0x00
    RIGHT = 90
    BACKWARD = 180
    LEFT = 270

class drone_error_type(int) :
    DOUBLE_TAKE_OFF = 0x00 #T
    DOUBLE_LAND = 0x01 #L
    PRE_FLIGHT_LANDING = 0x02 #D
    NO_IN_FLIGHT = 0x03 #N
    OTHER = 0x04 #O
    SUCCESS = 0xFF #S
    COMPLETE = 0xFE #C
 
class drone_command_param(int):
    HEADER = 0x54
    TAIL = 0x45

def wait_for_receive(serial_port:serial.Serial,size:int) -> bytes:
    buffer:bytes = b''
    while True:
        data = serial_port.read_all()
        buffer += data
        if len(buffer) >= size:
            return buffer
        time.sleep(0.01)

def take_off():
    command = bytearray([drone_command_param.HEADER,drone_command_type.TAKE_OFF,0x00,0x00,100,drone_command_param.TAIL])
    return command

def move(serial_port:serial.Serial,direction:int,distance:int):
    command = bytearray([
        drone_command_param.HEADER,
        drone_command_type.MOVE,
        int(direction/255),direction%255,distance,
        drone_command_param.TAIL
    ])
    return command

def land(serial_port:serial.Serial):
    command = bytearray([drone_command_param.HEADER,drone_command_type.LAND,0x00,0x00,0x00,drone_command_param.TAIL])
    return command

def error_get(character:int) -> drone_error_type:
    match (character):
        case 0x54: #T
            return drone_error_type.DOUBLE_TAKE_OFF
        case 0x4C: #L
            return drone_error_type.DOUBLE_LAND
        case 0x44: #D
            return drone_error_type.PRE_FLIGHT_LANDING
        case 0x4E: #N
            return drone_error_type.NO_IN_FLIGHT
        case 0x4F: #O
            return drone_error_type.OTHER
        case 0x53: #S
            return drone_error_type.SUCCESS
        case 0x43: #C
            return drone_error_type.COMPLETE
        case _:
            return drone_error_type.OTHER
    
def drone_command(serial_port:serial.Serial,type:drone_command_type,arg1:int,arg2:int):
    match (type):
        case drone_command_type.TAKE_OFF:
            command = take_off()
        case drone_command_type.MOVE:
            command = move(serial_port,arg1,arg2)
        case drone_command_type.LAND:
            command = land(serial_port)

    serial_port.write(command)
    command_state = serial_port.read(1)
    error = error_get(command_state[0])
    if error != drone_error_type.SUCCESS :
        return drone_result(False, f"Error: {error.name}")
    
    wait = wait_for_receive(serial_port, 1)
    if wait[0] != drone_error_type.COMPLETE:
        return drone_result(False, f"Error: UnKown")

    return drone_result(True, "Command executed successfully")


if __name__ == '__main__':
    ser = serial.Serial("/dev/ttyS3", 115200, timeout=1)
    drone_command(ser, drone_command_type.TAKE_OFF, 0, 0)
    # drone_command(ser, drone_command_type.MOVE, 0, 50)
    drone_command(ser, drone_command_type.LAND, 0, 0)

