from matplotlib import pyplot as plt
import numpy as np
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import rospy
from scipy import interpolate
import scipy
from ransac import*
import pickle
import transformations

RATE=1000

def interpolate_to_frequencies(rate, ori_time_list, pos_list, obj_end):
    frequency = rate
    time_list = []
    val_list = []
    obj_list = []
    for t in ori_time_list:
        time_in_sec = float(t.to_nsec() - ori_time_list[0].to_nsec()) / 1e9
        time_list.append(time_in_sec)
    for val in pos_list:
    	val_list.append(val)
    time_interval_obj = np.arange(0, obj_end, float(1) / frequency)
    f = interpolate.interp1d(time_list, val_list, kind='linear')
    val_list_obj = f(time_interval_obj)
    return val_list_obj

if __name__ == "__main__":
	with open("robot_2_21.txt", "r") as input_file:
		lines = input_file.readlines()
		time_stamp_mp = []
		pos_mp = []
		for line in lines:
			line = line.replace('\t', '').replace('\r', '')
			tmp = line.split("TimeStamp:")
		tmp = tmp[1:]
		for item in tmp:
			time_tmp, pos_tmp = item.split(" Position:")
			time_stamp_mp.append(rospy.Time.from_sec(float(time_tmp)))
			pos_mp.append(pos_tmp.split(','))
		pos_mp_x = []
		pos_mp_y = []
		pos_mp_z = []
		for item in pos_mp:
			pos_mp_x.append(float(item[0]))
			pos_mp_y.append(float(item[1]))
			pos_mp_z.append(float(item[2]))

	with open("position_calibration_new.txt", "r") as input_file2:
		lines = input_file2.readlines()
		time_stamp_rob = []
		pos_rob = []
		for line in lines:
			element = line.split(',')
			time_stamp_rob.append(rospy.Time.from_sec(float(element[0])))
			pos_rob.append([float(element[1]), float(element[2]), float(element[3])])

	pos_rob_x = [a[0] for a in pos_rob]
	pos_rob_y = [a[1] for a in pos_rob]
	pos_rob_z = [a[2] for a in pos_rob]
	
	start_stamp = 0
	end_stamp = 0
	flag = 0

	for idx in range(len(time_stamp_rob)):
		if time_stamp_mp[0] <= time_stamp_rob[idx] and flag == 0:
			start_stamp = time_stamp_rob[idx]
			flag = 1
	end_stamp = min(time_stamp_mp[-1], time_stamp_rob[-1])

	obj_end_stamp = float(end_stamp.to_nsec() - start_stamp.to_nsec()) / 1e9

	it_pos_rob_x = interpolate_to_frequencies(
								RATE, time_stamp_rob, pos_rob_x, obj_end_stamp)
	it_pos_rob_y = interpolate_to_frequencies(
								RATE, time_stamp_rob, pos_rob_y, obj_end_stamp)
	it_pos_rob_z = interpolate_to_frequencies(
								RATE, time_stamp_rob, pos_rob_z, obj_end_stamp)

	it_pos_mp_x = interpolate_to_frequencies(
								RATE, time_stamp_mp, pos_mp_x, obj_end_stamp)
	it_pos_mp_y = interpolate_to_frequencies(
								RATE, time_stamp_mp, pos_mp_y, obj_end_stamp)
	it_pos_mp_z = interpolate_to_frequencies(
								RATE, time_stamp_mp, pos_mp_z, obj_end_stamp)

	rob_traj = np.vstack((it_pos_rob_x, it_pos_rob_y, it_pos_rob_z)).T
	mp_traj = np.vstack((it_pos_mp_x, it_pos_mp_y, it_pos_mp_z)).T
	unit = np.ones(len(it_pos_rob_x))
	all_data = np.vstack((it_pos_mp_x, it_pos_mp_y, it_pos_mp_z, unit, it_pos_rob_x, it_pos_rob_y, it_pos_rob_z, unit)).T

	bias_mp_traj = np.zeros((len(mp_traj), 4))
	for i in range(len(mp_traj)):
		bias_mp_traj[i] = np.append(mp_traj[i], 1)

	trial_0 = np.vstack((it_pos_mp_x, it_pos_mp_y, it_pos_mp_z, unit)).T
	trial_1 = np.vstack((it_pos_rob_x, it_pos_rob_y, it_pos_rob_z, unit)).T

	trial_0 = np.transpose(trial_0)
	trial_1 = np.transpose(trial_1)
	BB = transformations.superimposition_matrix(trial_0, trial_1)
	print(BB)
	CC = numpy.dot(BB, trial_0)
	CC = np.transpose(CC)
	CC_x = [a[0] for a in CC]
	CC_y = [a[1] for a in CC]
	CC_z = [a[2] for a in CC]
	output = open("M_mr_rev.pkl", 'wb')
	pickle.dump(BB, output, -1)