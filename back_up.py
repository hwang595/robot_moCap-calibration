from usingRosBag import RosBagParser
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import Spacetime.arm as armFK
import numpy as np
import argparse
import rospy
from scipy import interpolate
import scipy
import pickle
import transformations

DEGREE_OF_FREEDOM = 6
NSEC_SEC_CONVERTER = 1e9
RATE = 1000

def readRobotReaderFile(fileName):
	with open(fileName, "r") as input_file2:
		lines = input_file2.readlines()
		time_stamp_rob = []
		pos_rob = []
		joint_angles = []
		for line in lines:
			element = line.split(',')
			time_stamp_rob.append(rospy.Time.from_sec(float(element[0])))
			pos_rob.append([float(element[1]), float(element[2]), float(element[3])])
			joint_angles.append(
				[float(element[4]), float(element[5]), float(element[6]), float(element[7]), float(element[8]), float(element[9])])
		return time_stamp_rob, pos_rob, joint_angles

def parsePosTables(posTable, special=False):
	#parse the pos table file (with time stamp contained) into three vectors
	#like [time_stamp] to vec_x, vec_y, vec_z
	if len(posTable[0]) == 4 and special == False:
		#data table from rosbag file
		stamp = [a[0] for a in posTable]
		vec_x = [a[1] for a in posTable]
		vec_y = [a[2] for a in posTable]
		vec_z = [a[3] for a in posTable]
		return np.array(stamp), np.array(vec_x), np.array(vec_y), np.array(vec_z)
	elif len(posTable[0]) == 3 or special == True:
		#data table from robot data reader
		vec_x = [a[0] for a in posTable]
		vec_y = [a[1] for a in posTable]
		vec_z = [a[2] for a in posTable]
		return np.array(vec_x), np.array(vec_y), np.array(vec_z)

def interpolate_to_frequencies(rate, ori_time_list, pos_list, obj_end):
	#interpolate to make signals from motion capture to robot at the same frequency 
    frequency = rate
    time_list = []
    val_list = []
    obj_list = []
    for t in ori_time_list:
        time_in_sec = float(t.to_nsec() - ori_time_list[0].to_nsec()) / NSEC_SEC_CONVERTER
        time_list.append(time_in_sec)
    for val in pos_list:
    	val_list.append(val)
    time_interval_obj = np.arange(0, obj_end, float(1) / frequency)
    f = interpolate.interp1d(time_list, val_list, kind='linear')
    val_list_obj = f(time_interval_obj)
    return val_list_obj

def get_endEffectorPos_with_jointAngle(joint_angles):
	arm = armFK.UR5()
	#let's just use the end effector position for current
	end_effector_pos = []
	for j_idx in range(len(joint_angles)):
		joint_state = joint_angles[j_idx]
		pts, frame = arm.getFrames(joint_state)
		end_effector_pos.append(pts[-1])
	return np.array(end_effector_pos)

def do_calib(rigidBodyPos, pos_rob, joint_angles, time_stamp_rob):
	#parse the position bag from the file for visualization
	moCap_time_stamp, moCap_pos_x, moCap_pos_y, moCap_pos_z = parsePosTables(rigidBodyPos)
	pos_rob_x, pos_rob_y, pos_rob_z = parsePosTables(pos_rob)

	start_stamp = max(rigidBodyPos[0][0], time_stamp_rob[0])
	end_stamp = min(rigidBodyPos[-1][0], time_stamp_rob[-1])
	obj_end_stamp = float(end_stamp.to_nsec() - start_stamp.to_nsec()) / NSEC_SEC_CONVERTER
	interpolated_joint_angle_table = [0] * DEGREE_OF_FREEDOM

	#interpolate to align signal from two data sources
	it_pos_rob_x = interpolate_to_frequencies(
								RATE, time_stamp_rob, pos_rob_x, obj_end_stamp)
	it_pos_rob_y = interpolate_to_frequencies(
								RATE, time_stamp_rob, pos_rob_y, obj_end_stamp)
	it_pos_rob_z = interpolate_to_frequencies(
								RATE, time_stamp_rob, pos_rob_z, obj_end_stamp)

	it_pos_mp_x = interpolate_to_frequencies(
								RATE, moCap_time_stamp, moCap_pos_x, obj_end_stamp)
	it_pos_mp_y = interpolate_to_frequencies(
								RATE, moCap_time_stamp, moCap_pos_y, obj_end_stamp)
	it_pos_mp_z = interpolate_to_frequencies(
								RATE, moCap_time_stamp, moCap_pos_z, obj_end_stamp)
	for i in range(len(joint_angles)):
		interpolated_joint_angle_table[i] = interpolate_to_frequencies(
								RATE, time_stamp_rob, joint_angles[i], obj_end_stamp)

	interpolated_joint_angle_table = np.transpose(interpolated_joint_angle_table)
	#plot test here:========================================================================
	plt.figure(2)
	plt.plot([i for i in range(len(interpolated_joint_angle_table))], 
			[a[0] for a in interpolated_joint_angle_table])
	plt.plot([i for i in range(len(interpolated_joint_angle_table))], 
			[a[1] for a in interpolated_joint_angle_table])
	plt.plot([i for i in range(len(interpolated_joint_angle_table))], 
			[a[2] for a in interpolated_joint_angle_table])
	plt.plot([i for i in range(len(interpolated_joint_angle_table))], 
			[a[3] for a in interpolated_joint_angle_table])
	plt.plot([i for i in range(len(interpolated_joint_angle_table))], 
			[a[4] for a in interpolated_joint_angle_table])
	plt.plot([i for i in range(len(interpolated_joint_angle_table))], 
			[a[5] for a in interpolated_joint_angle_table])													
	#=======================================================================================	
	rob_endEffector_pos = get_endEffectorPos_with_jointAngle(interpolated_joint_angle_table)
	rob_EE_pos_x, rob_EE_pos_y, rob_EE_pos_z = parsePosTables(rob_endEffector_pos)

	#unit = np.ones(len(it_pos_rob_x))
	unit = np.ones(len(rob_EE_pos_x))
	trial_moCap = np.vstack((it_pos_mp_x, it_pos_mp_y, it_pos_mp_z, unit)).T
	trial_robot = np.vstack((rob_EE_pos_x, rob_EE_pos_y, rob_EE_pos_z, unit)).T
	#trial_robot = np.vstack((it_pos_rob_x, it_pos_rob_y, it_pos_rob_z, unit)).T

	#calc transformation matrix between moCap data to robot data
	trial_moCap = np.transpose(trial_moCap)
	trial_robot = np.transpose(trial_robot)

	M_mocap_robot = transformations.superimposition_matrix(trial_moCap, trial_robot)

	matrix_after_transform = np.dot(M_mocap_robot, trial_moCap)
	matrix_after_transform = np.transpose(matrix_after_transform)
	return matrix_after_transform, M_mocap_robot, rob_endEffector_pos

def transform_orientation(quaternion, trans_matrix):
	MVR = np.zeros((3, 3))
	PMR = np.array([])
	for r_idx in range(len(MVR_226)-1):
	    row = MVR_226[r_idx]
	    MVR[r_idx] = row[0:3]
	    PMR = np.append(PMR, row[3])

	for q_idx in range(len(quaternion)):
	    q_for_tran = quaternion[q_idx]
	    t_for_tran = transformations.quaternion_matrix_rev(q_for_tran)
	    tmp_M_1 = np.dot(MVR, t_for_tran)
	    transformed_q =  transformations.quaternion_from_matrix(tmp_M_1)
	    ret_quat[q_idx] = transformed_q
	return ret_quat

if __name__ == '__main__':
	start_stamp = 0
	end_stamp = 0
	flag = 0
	#include and parse ros-bag data, include the rigide body position and orientation
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', help='The file name/path of the bag.', required=True)
	args = vars(parser.parse_args())
	bagname = args["filename"]
	bag_data = RosBagParser(resampleRate=1000)
	rigidBodyPos, rigidBodyQuat = bag_data.parseTongsBag(bagname)

	#load robot position data and parse it to get robot position matrix
	fileName = "calibration_final_round.txt"
	time_stamp_rob, pos_rob, joint_angles = readRobotReaderFile(fileName)

	pos_rob_x, pos_rob_y, pos_rob_z = parsePosTables(pos_rob)
	#reshape the joint angles to make it easier to interpolate
	reshaped_joint_angles = np.transpose(np.array(joint_angles))
	moCap_time_stamp, moCap_pos_x, moCap_pos_y, moCap_pos_z = parsePosTables(rigidBodyPos)
	matrix_after_transform, M_mocap_robot, rob_endEffector_pos = do_calib(rigidBodyPos, pos_rob, reshaped_joint_angles, time_stamp_rob)
	rob_EE_pos_x, rob_EE_pos_y, rob_EE_pos_z = parsePosTables(rob_endEffector_pos)
	CC_x, CC_y, CC_z = parsePosTables(matrix_after_transform, special=True)

	#some debug code here:
	'''
	print(start_stamp, end_stamp)
	print(float(end_stamp.to_nsec() - start_stamp.to_nsec()) / NSEC_SEC_CONVERTER)
	for item_idx in range(len(time_stamp_rob)):
		if item_idx <= 5:
			print(time_stamp_rob[item_idx])
	
	print('=============================================================================')
	for item_idx in range(len(rigidBodyPos)):
		if item_idx <= 5:
			print(rigidBodyPos[item_idx][0].to_nsec())
	'''
	#do some visulization here:
	fig = plt.figure(1)
	ax = fig.add_subplot(111, projection='3d')
	#robot trajectory
	ax.plot(pos_rob_x, pos_rob_y, pos_rob_z, c='b', linewidth=1)
	#moCap transformed data (this should be aligned with the robot data)
	ax.plot(rob_EE_pos_x, rob_EE_pos_y, rob_EE_pos_z, c='r', linewidth=1)
#	ax.plot(CC_x, CC_y, CC_z, c='r', linewidth=1)
#	ax.plot(moCap_pos_x, moCap_pos_y, moCap_pos_z, c='g', linewidth=1)

	plt.show()