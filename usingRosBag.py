#coding utf 8 -*-
#Author: hwang
#Version:3.0.0
#Created on: Jan 15, 2017
#Revised on:
import rosbag
import argparse
from tf import TransformerROS as rtf
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import math
#from urdf_parser_py.urdf import URDF
#import roslib; roslib.load_manifest('urdf_parser_py')

NSEC_SEC_CONVERTER = 1e9

class RosBagParser:

    def __init__(self, resampleRate=None):
        self.resampleRate = resampleRate
        self.forceSensor1array_interpolated = []
        self.forceSensor2array_interpolated = []
        self.forceSensor1array_x_interpolated = []
        self.forceSensor1array_y_interpolated = []
        self.forceSensor1array_z_interpolated = []
        self.forceSensor2array_x_interpolated = []
        self.forceSensor2array_y_interpolated = []
        self.forceSensor2array_z_interpolated = [] 
        self.encoderarray_interpolated = []
        self.vivePos_interpolated = []
        self.viveQuat_interpolated = []
        self.rigidBodyPos_interpolated = []
        self.rigidBodyQuat_interpolated = []
        self.resample_time_stamp = []
        self.topics = ["encoder", "forceSensor1_vecx", "forceSensor1_vecy", 
        "forceSensor1_vecz", "forceSensor2_vecx", "forceSensor2_vecy", "forceSensor2_vecz", "ViveQuat", "VivePos"]
        self.original_sampleRate = {}

    def interpolate_to_frequencies(self, ros_bag_list, global_end_time):
        frequency = self.resampleRate
        time_list = []
        val_list = []
        obj_list = []
        for item in ros_bag_list:
            time_in_sec = float(item[0].to_nsec() - ros_bag_list[0][0].to_nsec()) / 1e9
            time_list.append(time_in_sec)
            val_list.append(item[1])
        time_interval_obj = np.arange(0, global_end_time, float(1) / frequency)
        f = interpolate.interp1d(time_list, val_list, kind='linear')
        val_list_obj = f(time_interval_obj)
        return val_list_obj

    #handle those vectors like forceSensors and vive_positions in 3d
    def expand_list_3d(self, ros_bag_list, time_stamp=None):
        if len(ros_bag_list[0]) == 4:
            vector_x = [[item[0], item[1]] for item in ros_bag_list]
            vector_y = [[item[0], item[2]] for item in ros_bag_list]
            vector_z = [[item[0], item[3]] for item in ros_bag_list]
        elif len(ros_bag_list[0]) == 3:
        	vector_x = []
        	vector_y = []
        	vector_z = []
        	for i in range(len(ros_bag_list)):
        		vector_x.append([time_stamp[i], ros_bag_list[i][0]])
        		vector_y.append([time_stamp[i], ros_bag_list[i][1]])
        		vector_z.append([time_stamp[i], ros_bag_list[i][2]])
        return vector_x, vector_y, vector_z

    def expand_list_4d(self, ros_bag_list):
    	vector_w = []
    	vector_x = []
    	vector_y = []
    	vector_z = []
        vector_stamp = []
        if len(ros_bag_list[0]) == 4:
            for i in range(len(ros_bag_list)):
                vector_w.append(ros_bag_list[i][0])
                vector_x.append(ros_bag_list[i][1])
                vector_y.append(ros_bag_list[i][2])
                vector_z.append(ros_bag_list[i][3])
            return vector_w, vector_x, vector_y, vector_z
        elif len(ros_bag_list[0]) == 5:
            vector_stamp = [item[0] for item in ros_bag_list]
            vector_w = [item[1] for item in ros_bag_list]
            vector_x = [item[2] for item in ros_bag_list]
            vector_y = [item[3] for item in ros_bag_list]
            vector_z = [item[4] for item in ros_bag_list]
            return vector_stamp, vector_w, vector_x, vector_y, vector_z

    def interpolate_with_integration(self, vec_x, vec_y, vec_z, global_end_time):
        frequency = self.resampleRate
        vec_x_inter = []
        vec_y_inter = []
        vec_z_inter = []
        integrated_list = []
        vec_x_inter = self.interpolate_to_frequencies(vec_x, global_end_time)
        vec_y_inter = self.interpolate_to_frequencies(vec_y, global_end_time)
        vec_z_inter = self.interpolate_to_frequencies(vec_z, global_end_time)
        zip_val = zip(vec_x_inter, vec_y_inter, vec_z_inter)
        for z in zip_val:
            integrated_list.append(list(z))
        return integrated_list

    def convert_msgsVec3_to_list(self, msgsVec3_list):
    	ret_list = []
    	for item in msgsVec3_list:
    		ret_list.append([item.x, item.y, item.z])
    	return ret_list

    def convert_quart_to_list(self, quart_list):
    	ret_list = []
    	for item in quart_list:
    		ret_list.append([item.w, item.x, item.y, item.z])
    	return ret_list

    def unit_vector(self, data, axis=None, out=None):
        """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3))
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1]))
        [1.0]

        """
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
                return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
            data = out
        length = np.atleast_1d(np.sum(data*data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data

    def quaternion_slerp(self, quat0, quat1, fraction, spin=0, shortestpath=True):
        """Return spherical linear interpolation between two quaternions.

        >>> q0 = random_quaternion()
        >>> q1 = random_quaternion()
        >>> q = quaternion_slerp(q0, q1, 0)
        >>> numpy.allclose(q, q0)
        True
        >>> q = quaternion_slerp(q0, q1, 1, 1)
        >>> numpy.allclose(q, q1)
        True
        >>> q = quaternion_slerp(q0, q1, 0.5)
        >>> angle = math.acos(numpy.dot(q0, q))
        >>> numpy.allclose(2, math.acos(numpy.dot(q0, q1)) / angle) or \
            numpy.allclose(2, math.acos(-numpy.dot(q0, q1)) / angle)
        True

        """
        _EPS = np.finfo(float).eps * 4.0
        q0 = self.unit_vector(quat0[:4])
        q1 = self.unit_vector(quat1[:4])
        if fraction == 0.0:
            return q0
        elif fraction == 1.0:
            return q1
        d = np.dot(q0, q1)
        if abs(abs(d) - 1.0) < _EPS:
            return q0
        if shortestpath and d < 0.0:
            # invert rotation
            d = -d
            np.negative(q1, q1)
        angle = math.acos(d) + spin * math.pi
        if abs(angle) < _EPS:
            return q0
        isin = 1.0 / math.sin(angle)
        q0 *= math.sin((1.0 - fraction) * angle) * isin
        q1 *= math.sin(fraction * angle) * isin
        q0 += q1
        return q0

    def quart_slerp_interpolate(self, quart_w, quart_x, quart_y, quart_z, time_stamp, global_end_time):
        frequency = self.resampleRate
        de_bug_counter = 0
        ret_list = []
        temp_quart = []
        time_interval_obj = np.arange(
                        0, global_end_time, float(1)/frequency)
        iter_op_0 = zip(quart_w, quart_x, quart_y, quart_z)
        for it_0 in iter_op_0:
            temp_quart.append(it_0)
        range_counter = 0
        i_counter = 0
        for t_idx in range(len(time_interval_obj)):
            time_comp = round(float(time_stamp[range_counter].to_nsec() - time_stamp[0].to_nsec()) / NSEC_SEC_CONVERTER, 3)
            last_time_comp = round(float(time_stamp[range_counter-1].to_nsec() - time_stamp[0].to_nsec()) / NSEC_SEC_CONVERTER, 3)
            if time_interval_obj[t_idx] < time_comp:
                fraction = (float(time_interval_obj[t_idx]) - last_time_comp) / (time_comp - last_time_comp)
                ret_list.append(self.quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
            elif (time_interval_obj[t_idx] - time_comp) < 1e-8:       
                range_counter += 1
                fraction = 0
                if range_counter == len(temp_quart):
                    ret_list.append(temp_quart[-1])
                    return ret_list
                ret_list.append(self.quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
            else:
                while time_interval_obj[t_idx] > time_comp:
                    range_counter += 1
                    time_comp = round(float(time_stamp[range_counter].to_nsec() - time_stamp[0].to_nsec()) / NSEC_SEC_CONVERTER, 3)
                ret_list.append(self.quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
        return ret_list

    def detect_if_emtpy(sefl, array_table):
        #seq in this empty_list is quite important
        #for current version, the seq is assigned as follows:
        #[encoderarray, forceSensor1array, forceSensor2array, ... 
        # ... viveQuat, vivePos, rigidBodyPos, rigidBodyQuat]
        #former processing and interpolation are based on this seq
        empty_list = [0] * len(array_table)
        for array_row_idx in range(len(array_table)):
            if len(array_table[array_row_idx]) == 0:
                empty_list[array_row_idx] = 1
        return empty_list

    def calculate_global_end_time(
        self, empty_flag_list, encoderarray, forceSensor1array, forceSensor2array, rigidBodyPos, rigidBodyQuat, time_quat, time_pos):
        end_time_list = []
        if empty_flag_list[0] == 0:
            encoder_end_time = float(encoderarray[-1][0].to_nsec() - encoderarray[0][0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['encoder_rate'] = len(encoderarray) / float(encoder_end_time)
            end_time_list.append(encoder_end_time)
        if empty_flag_list[1] == 0:
            force1_end_time = float(forceSensor1array[-1][0].to_nsec() - forceSensor1array[0][0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['force_sensor1_rate'] = len(forceSensor1array) / float(force1_end_time)
            end_time_list.append(force1_end_time)
        if empty_flag_list[2] == 0:
            force2_end_time = float(forceSensor2array[-1][0].to_nsec() - forceSensor2array[0][0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['force_sensor2_rate'] = len(forceSensor2array) / float(force2_end_time)
            end_time_list.append(force2_end_time)
        if empty_flag_list[3] == 0:
            quat_end_time = float(time_quat[-1].to_nsec() - time_quat[0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['quaternion_rate'] = len(time_quat) / float(quat_end_time)
            end_time_list.append(quat_end_time)
        if empty_flag_list[4] == 0:
            pos_end_time = float(time_pos[-1].to_nsec() - time_pos[0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['position_rate'] = len(time_pos) / float(pos_end_time)
            end_time_list.append(pos_end_time)
        if empty_flag_list[5] == 0:
            #rbp stands for rigid body position
            rbp_end_time = float(rigidBodyPos[-1][0].to_nsec() - rigidBodyPos[0][0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['rigidBodyPos_rate'] = len(rigidBodyPos) / float(rbp_end_time)
            end_time_list.append(rbp_end_time)
        if empty_flag_list[6] == 0:
            #rbq stands for rigid body orientation
            rbq_end_time = float(rigidBodyQuat[-1][0].to_nsec() - rigidBodyQuat[0][0].to_nsec()) / NSEC_SEC_CONVERTER 
            self.original_sampleRate['rigidBodyQuat_rate'] = len(rigidBodyQuat) / float(rbq_end_time)
            end_time_list.append(rbq_end_time)            
        return min(end_time_list)

    def interpolate_encoder_signal(self, encoderarray, global_end_time):
        self.encoderarray_interpolated = self.interpolate_to_frequencies(encoderarray, global_end_time)

    def interpolate_force_signal(self, forceSensor_array, global_end_time, forceSensor_num):
        force_vec_x, force_vec_y, force_vec_z = self.expand_list_3d(forceSensor_array)
        if forceSensor_num == '1':
            self.forceSensor1array_interpolated = self.interpolate_with_integration(
                                                force_vec_x, force_vec_y, force_vec_z, global_end_time)
        if forceSensor_num == '2':
            self.forceSensor2array_interpolated = self.interpolate_with_integration(
                                                force_vec_x, force_vec_y, force_vec_z, global_end_time)

    def interpolate_quat_signal(self, viveQuat, time_quat, global_end_time):
        quart_vec_w, quart_vec_x, quart_vec_y, quart_vec_z = self.expand_list_4d(self.convert_quart_to_list(viveQuat))
        self.viveQuat_interpolated = self.quart_slerp_interpolate(
                                            quart_vec_w, quart_vec_x, quart_vec_y, quart_vec_z, time_quat, global_end_time)

    def interpolate_pos_signal(self, vivePos, time_pos, global_end_time):
        pos_vec_x, pos_vec_y, pos_vec_z = self.expand_list_3d(self.convert_msgsVec3_to_list(vivePos), time_pos)
        self.vivePos_interpolated = self.interpolate_with_integration(
                                            pos_vec_x, pos_vec_y, pos_vec_z, global_end_time)

    def interpolate_rb_pos_signal(self, rbPos, global_end_time):
        rb_pos_x, rb_pos_y, rb_pos_z = self.expand_list_3d(rbPos)
        self.rigidBodyPos_interpolated = self.interpolate_with_integration(
                                            rb_pos_x, rb_pos_y, rb_pos_z, global_end_time)

    def interpolate_rb_quat_signal(self, rbQuat, global_end_time):
        rb_quart_time, rb_quart_w, rb_quart_x, rb_quart_y, rb_quart_z = self.expand_list_4d(rbQuat)
        self.rigidBodyQuat_interpolated = self.quart_slerp_interpolate(
                                            rb_quart_w, rb_quart_x, rb_quart_y, rb_quart_z, rb_quart_time, global_end_time)

    def parseTongsBag(self, bagname):
        #we make several flags here to make sure if one or several channels of signal are empty
        encoder_is_empty = 0 
        force1_is_empty = 0
        force2_is_empty = 0
        viveQuat_is_empty = 0
        vivePos_is_empty = 0
        rigidBodyPos_is_empty = 0
        rigidBodyQuat_is_empty = 0
        empty_flag_list = [encoder_is_empty, force1_is_empty, force2_is_empty,
                            viveQuat_is_empty, vivePos_is_empty, rigidBodyPos_is_empty,
                            rigidBodyQuat_is_empty]

        bag = rosbag.Bag(bagname)
        #arrays tp restore data from rosbags
        forceSensor1array = []
        forceSensor2array = []
        encoderarray = []
        vivePos = []
        viveQuat = []
        rigidBodyPos = []
        rigidBodyQuat = []
        time_pos = [] #time_stamp for position values
        time_quat = [] #time_stamp for quaternion values
        #arrays to restore interpolated rosbags
        time_encoder = []
        topics = set()
        for topic, msg, t in bag.read_messages():
            topics.add(topic)
            if (topic == '/tongs/encoder'):
                time_encoder.append(t)
                encoderarray.append([msg.header.stamp, msg.position[0]])
            elif (topic == '/forceSensor1'):
                forceSensor1array.append([msg.header.stamp, msg.vector.x, msg.vector.y, msg.vector.z])
            elif (topic == '/forceSensor2'):
                forceSensor2array.append([msg.header.stamp, msg.vector.x, msg.vector.y, msg.vector.z])
            elif (topic == '/ViveQuat'):
                time_quat.append(t)
                viveQuat.append(msg)
            elif (topic == '/VivePos'):
                time_pos.append(t)
                vivePos.append(msg)
            elif (topic == '/Robot_1/pose'):
                rigidBodyPos.append([msg.header.stamp, msg.pose.position.x,
                                    msg.pose.position.y, msg.pose.position.z])
                rigidBodyQuat.append([msg.header.stamp, msg.pose.orientation.w,
                                    msg.pose.orientation.x, msg.pose.orientation.y,
                                    msg.pose.orientation.z])
            else:
                pass
        #merge signal arrays into a table to detect if any of them is empty        
        array_table = [encoderarray, forceSensor1array, forceSensor2array, 
            viveQuat, vivePos, rigidBodyPos, rigidBodyQuat]

        empty_flag_list = self.detect_if_emtpy(array_table)

        global_end_time = self.calculate_global_end_time(
            empty_flag_list, encoderarray, forceSensor1array, 
            forceSensor2array, rigidBodyPos, rigidBodyQuat, time_quat, time_pos)
        #Please start from here next time
        #================================
        #================================
        global_start_time = 0
        synchronous_time_list = np.arange(global_start_time, global_end_time, float(1) / self.resampleRate)
        self.resample_time_stamp = synchronous_time_list

        #start interpolation for encoder, force sensors, and VivePos data
        if empty_flag_list[0] == 0:
            self.interpolate_encoder_signal(encoderarray, global_end_time)
        #interpolating force data 1 and force data 2:
        if empty_flag_list[1] == 0:
            self.interpolate_force_signal(forceSensor1array, global_end_time, '1')
            self.forceSensor1array_x_interpolated = [force1[0] for force1 in self.forceSensor1array_interpolated]
            self.forceSensor1array_y_interpolated = [force1[1] for force1 in self.forceSensor1array_interpolated]
            self.forceSensor1array_z_interpolated = [force1[2] for force1 in self.forceSensor1array_interpolated]
        if empty_flag_list[2] == 0:
            self.interpolate_force_signal(forceSensor2array, global_end_time, '2')
            self.forceSensor2array_x_interpolated = [force2[0] for force2 in self.forceSensor2array_interpolated]
            self.forceSensor2array_y_interpolated = [force2[1] for force2 in self.forceSensor2array_interpolated]
            self.forceSensor2array_z_interpolated = [force2[2] for force2 in self.forceSensor2array_interpolated]
        if empty_flag_list[3] == 0:
            self.interpolate_quat_signal(viveQuat, time_quat, global_end_time)
        if empty_flag_list[4] == 0:
            self.interpolate_pos_signal(vivePos, time_pos, global_end_time)
        if empty_flag_list[5] == 0:
            self.interpolate_rb_pos_signal(rigidBodyPos, global_end_time)
        if empty_flag_list[6] == 0:
            self.interpolate_rb_quat_signal(rigidBodyQuat, global_end_time)
        bag.close()

        return rigidBodyPos, rigidBodyQuat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', help='The file name/path of the bag.', required=True)
    args = vars(parser.parse_args())
    bagname = args["filename"]
    bag_data = RosBagParser(resampleRate=1000)
    _, _ = bag_data.parseTongsBag(bagname)

    #plotting the encoder data
    '''
    if empty_flag_list[0] == 0:
        plt.figure(1)
        plt.title('Encoder')
        plt.plot(bag_data.resample_time_stamp, bag_data.encoderarray_interpolated, 'b^')

    if empty_flag_list[1] == 0:
        plt.figure(2)
        plt.subplot(311)
        plt.title('x-axis')
        plt.plot(bag_data.resample_time_stamp, [a[0] for a in bag_data.forceSensor1array_interpolated], 'r^')
        plt.subplot(312)
        plt.title('y-axis')
        plt.plot(bag_data.resample_time_stamp, [a[1] for a in bag_data.forceSensor1array_interpolated], 'g^')
        plt.subplot(313)
        plt.title('z-axis')
        plt.plot(bag_data.resample_time_stamp, [a[2] for a in bag_data.forceSensor1array_interpolated], 'b^')

    if empty_flag_list[2] == 0:
        plt.figure(3)
        plt.subplot(311)
        plt.title('x-axis')
        plt.plot(bag_data.resample_time_stamp, [a[0] for a in bag_data.forceSensor2array_interpolated], 'r^')
        plt.subplot(312)
        plt.title('y-axis')
        plt.plot(bag_data.resample_time_stamp, [a[1] for a in bag_data.forceSensor2array_interpolated], 'g^')
        plt.subplot(313)
        plt.title('z-axis')
        plt.plot(bag_data.resample_time_stamp, [a[2] for a in bag_data.forceSensor2array_interpolated], 'b^')

    if empty_flag_list[3] == 0:
    #plotting the vive_position data
        plt.figure(4)
        plt.subplot(311)
        plt.title('x-axis')
        plt.plot(bag_data.resample_time_stamp, [a[0] for a in bag_data.vivePos_interpolated], 'r^')
        plt.subplot(312)
        plt.title('y-axis')
        plt.plot(bag_data.resample_time_stamp, [a[1] for a in bag_data.vivePos_interpolated], 'g^')
        plt.subplot(313)
        plt.title('z-axis')
        plt.plot(bag_data.resample_time_stamp, [a[2] for a in bag_data.vivePos_interpolated], 'b^')
        #plotting the vive_quaternion data
    if empty_flag_list[4] == 0:
        plt.figure(5)
        plt.subplot(411)
        plt.title('w')
        plt.plot(bag_data.resample_time_stamp, [a[0] for a in bag_data.viveQuat_interpolated], 'y^')
        plt.subplot(412)
        plt.title('x')
        plt.plot(bag_data.resample_time_stamp, [a[1] for a in bag_data.viveQuat_interpolated], 'r^')
        plt.subplot(413)
        plt.title('y')
        plt.plot(bag_data.resample_time_stamp, [a[2] for a in bag_data.viveQuat_interpolated], 'g^')
        plt.subplot(414)
        plt.title('z')
        plt.plot(bag_data.resample_time_stamp, [a[3] for a in bag_data.viveQuat_interpolated], 'b^')  
    plt.show()
    '''