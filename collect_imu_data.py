# usage: python3 stream_gyro_packed.py [mac1] [mac2] ... [mac(n)]
from __future__ import print_function

import mbientlab
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event

import csv
import os
import argparse
import platform
import sys
from datetime import date, datetime

# test subject id, trial id, timestamp

parser = argparse.ArgumentParser(description='IMU data collection.')
parser.add_argument('--sub_id', type=str, help='subject id (000, 001, 002)')
parser.add_argument('--pose', type=str, help='posture type (good, mild, bad)')
parser.add_argument('--trial_id', type=str, help='trial id (000, 001, 002)')
args = parser.parse_args()

if not args.sub_id:
    print("Please enter the subject id")
    raise Exception("Empty subject id")

if not args.sub_id:
    print("Please enter the trial id")
    raise Exception("Empty trial id")
    
if not args.pose:
    print("Please enter the posture type (good, mild, bad)")
    raise Exception("Missing posture type")

DATA_DIR = f"../data/{args.pose}"

ts = date.today()
now = datetime.now()
data_filename_suffix = f"s_{args.sub_id}_t_{args.trial_id}}.csv"
acc_data_filename = "acc_" + data_filename_suffix
gyro_data_filename = "gyro_" + data_filename_suffix
mag_data_filename = "mag_"+data_filename_suffix

filenames = [acc_data_filename, gyro_data_filename, mag_data_filename]

if sys.version_info[0] == 2:
    range = xrange


def check_data_conflicts() -> bool:
    existing_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    for file in existing_files:
        filename = os.path.split(file)[-1]
        t_start_ndx = filename.find('_t_')
        s_start_ndx = filename.find('_s_')

        # NOTE: The trial id string size is assumed to be 3.
        # If the size changes, please change 6 to reflect the new size.
        trial_id = filename[t_start_ndx + 3:t_start_ndx + 6]
        sub_id = filename[s_start_ndx + 3:s_start_ndx + 6]

        if args.sub_id == sub_id and args.trial_id == trial_id:
            raise Exception(f"Data with the subject id {args.sub_id} and trial id {args.trial_id} already exists.")

    return True


class State:
    # init
    def __init__(self, device):
        self.device = device
        self.samples = 0
        self.accCallback = FnVoid_VoidP_DataP(self.acc_data_handler)
        self.gyroCallback = FnVoid_VoidP_DataP(self.gyro_data_handler)
        self.magCallback = FnVoid_VoidP_DataP(self.mag_data_handler)
        self.acc_data_filename = ''
        self.gyro_data_filename = ''
        self.mag_data_filename = ''
    
    def name_make(self):
        # do self to make sure it is global inside class
        fn, fn_ext = os.path.splitext(acc_data_filename)
        fn += "_" + self.device.address
        self.acc_data_filename = fn + fn_ext

        fn, fn_ext = os.path.splitext(gyro_data_filename)
        fn += "_"+self.device.address
        self.gyro_data_filename = fn + fn_ext

        fn, fn_ext = os.path.splitext(mag_data_filename)
        fn += "_"+self.device.address
        self.mag_data_filename = fn + fn_ext
                
    def acc_data_handler(self, ctx, data):
        values = parse_value(data)

        # save file
        with open(f'../data/{args.pose}/{self.acc_data_filename}', "a", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow([data.contents.epoch, values.x, values.y, values.z])
        
        print("ACCEL: %s -> epoch: %s, data: %s" % (self.device.address, data.contents.epoch, values))
        self.samples+= 1

    def gyro_data_handler(self, ctx, data):
        # gyro callback
        values = parse_value(data)
        
        with open(f'../data/{args.pose}/{self.gyro_data_filename}', "a", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow([data.contents.epoch, values.x, values.y, values.z])
        
        print("GYRO: %s -> epoch: %s, data: %s" % (self.device.address, data.contents.epoch, values))
        self.samples+= 1

    def mag_data_handler(self, ctx, data):
       
        values = parse_value(data)
        with open(f'../data/{args.pose}/{self.mag_data_filename}', "a", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow([data.contents.epoch, values.x, values.y, values.z])
         
        print("MAG: %s -> %s" % (self.device.address, parse_value(data)))
        self.samples+= 1

# Check if the input subject and trial ids overwrite the existing data.
if check_data_conflicts():
    print("No data naming conflict detected")

states = []
device_ips = ["CA:C5:44:E0:3B:C3", "FF:EB:CA:C9:92:CF", "DF:D6:82:88:AF:42"]
# connect
# If the device is not responding, then restart the bluetooth service
# by running 'sudo /etc/init.d/bluetooth restart'
# instead of restart, do stop. then do start
for ndx, i in enumerate([0,1,2]):
    d = MetaWear(device_ips[i])
    d.disconnect()
    
    while True:
        try:
            d.disconnect()
            d.connect()
            print("Connected to " + d.address + " over " + ("USB" if d.usb.is_connected else "BLE"))
            break
        except mbientlab.warble.WarbleException:
            print("Retrying to connect...")
            continue
    
    states.append(State(d))
    states[ndx].name_make()


for s in states:
    print("Configuring device")
    libmetawear.mbl_mw_settings_set_connection_parameters(s.device.board, 7.5, 7.5, 0, 6000)
    sleep(1.5)
    
    # setup acc
    print("Configuring acc")
    libmetawear.mbl_mw_acc_bmi270_set_odr(s.device.board, AccBmi270Odr._50Hz);
    libmetawear.mbl_mw_acc_bosch_set_range(s.device.board, AccBoschRange._4G);
    libmetawear.mbl_mw_acc_write_acceleration_config(s.device.board);
    
    # setup gyro
    print("Configuring acc")
    libmetawear.mbl_mw_gyro_bmi270_set_range(s.device.board, GyroBoschRange._1000dps);
    libmetawear.mbl_mw_gyro_bmi270_set_odr(s.device.board, GyroBoschOdr._50Hz);
    libmetawear.mbl_mw_gyro_bmi270_write_config(s.device.board);

    #setup mag
    libmetawear.mbl_mw_mag_bmm150_stop(s.device.board)
    libmetawear.mbl_mw_mag_bmm150_set_preset(s.device.board, MagBmm150Preset.REGULAR)
   
    # get acc and subscribe
    acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_subscribe(acc, None, s.accCallback)

    # get gyro and subscribe
    print("Packed signal")
    gyro = libmetawear.mbl_mw_gyro_bmi270_get_packed_rotation_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_subscribe(gyro, None, s.gyroCallback)

    # get mag and subscribe
    mag = libmetawear.mbl_mw_mag_bmm150_get_b_field_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_subscribe(mag, None, s.magCallback)

    # start acc
    print("Start Acc")
    libmetawear.mbl_mw_acc_enable_acceleration_sampling(s.device.board)
    libmetawear.mbl_mw_acc_start(s.device.board)
    
    # start gyro
    print("Start gyro")
    libmetawear.mbl_mw_gyro_bmi270_enable_rotation_sampling(s.device.board)
    libmetawear.mbl_mw_gyro_bmi270_start(s.device.board)

    # start mag
    print("Start mag")
    libmetawear.mbl_mw_mag_bmm150_enable_b_field_sampling(s.device.board)
    libmetawear.mbl_mw_mag_bmm150_start(s.device.board)

# sleep
sleep(30.0)

# stop
for s in states:
    libmetawear.mbl_mw_mag_bmm150_stop(s.device.board)
    libmetawear.mbl_mw_mag_bmm150_disable_b_field_sampling(s.device.board)

    mag = libmetawear.mbl_mw_mag_bmm150_get_b_field_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(mag)

    libmetawear.mbl_mw_acc_stop(s.device.board)
    libmetawear.mbl_mw_acc_disable_acceleration_sampling(s.device.board)
    
    libmetawear.mbl_mw_gyro_bmi270_stop(s.device.board)
    libmetawear.mbl_mw_gyro_bmi270_disable_rotation_sampling(s.device.board)
    
    acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(acc)

    gyro = libmetawear.mbl_mw_gyro_bmi270_get_packed_rotation_data_signal(s.device.board)
    libmetawear.mbl_mw_datasignal_unsubscribe(gyro)
    
    libmetawear.mbl_mw_debug_disconnect(s.device.board)

# recap
print("Total Samples Received")
for s in states:
    print("%s -> %d" % (s.device.address, s.samples))
