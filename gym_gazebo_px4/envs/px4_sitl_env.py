'''
Author: Lei He
Date: 2021-04-15 10:17:06
LastEditTime: 2022-12-13 10:56:25
Description: 
Github: https://github.com/heleidsn
'''
import torch as th
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random

import os
import math
import rospy
import rospkg
import cv2
from .gazebo_connection import GazeboConnection

from cv_bridge import CvBridge, CvBridgeError

from mavros_msgs.msg import State, PositionTarget
from sensor_msgs.msg import NavSatFix
from mavros_msgs.srv import ParamSet, SetMode, CommandBool, CommandTOL, CommandHome
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from mavros_msgs.srv import CommandHome, CommandLong
from sensor_msgs.msg import Image
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from td3fd.msg import TrainInfo
from td3fd.msg import FeatureMaps

from visualization_msgs.msg import Marker

import subprocess32 as subprocess



class PX4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        rospy.logdebug('start init mavdrone_test_env_helei node')
        print('init px4 env...')
        # -----------------init ros node--------------------------------------
        rospy.init_node('mavdrone_test_env_helei', anonymous=False, log_level=rospy.INFO)
        
        self.gazebo = GazeboConnection(start_init_physics_parameters=False, reset_world_or_sim="WORLD")
        self.seed()

        self.px4_ekf2_path = os.path.join(rospkg.RosPack().get_path("px4"),"build/px4_sitl_default/bin/px4-ekf2")
        self.bridge = CvBridge()

        self._rate = rospy.Rate(10.0)

        self.max_depth_meter_gazebo = 18

        self.model = None
        self.use_ppo = False  # check if use ppo or other algos, used only for visualization
        self.policy = None
        self.obs_type = 'vector'  # vector or image
        self.use_expert = False

        # Subscribers 
        rospy.Subscriber('mavros/state', State, callback=self._stateCb, queue_size=10)
        rospy.Subscriber('mavros/local_position/pose', PoseStamped , callback=self._poseCb, queue_size=10)
        rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, callback=self._local_vel_Cb, queue_size=10)
        rospy.Subscriber('/mavros/global_position/raw/fix', NavSatFix, callback=self._gpsCb, queue_size=10)
        rospy.Subscriber('/camera/depth/image_raw', Image, callback=self._image_gazebo_Cb, queue_size=10)
        rospy.Subscriber('/mavros/setpoint_position/local_avoidance', PoseStamped, callback=self._pose_avoidanceCb, queue_size=10)
        rospy.Subscriber('/mavros/local_position/odom', Odometry, callback=self._local_odomCb, queue_size=10)

        # Publishers
        self._local_vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_test',TwistStamped, queue_size=10)
        self._local_pose_setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local',PoseStamped, queue_size=10)
        self._goal_pose_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self._action_velocity_body_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
        self._train_info_pub = rospy.Publisher('/environment/train_info', TrainInfo, queue_size=10)
        self.feature_maps_pub = rospy.Publisher('/environment/feature_maps', FeatureMaps, queue_size=10)
        self._mavros_cmd_client = rospy.ServiceProxy('/mavros/cmd/command',
                                                     CommandLong)

        # image
        self._depth_image_gray_input = rospy.Publisher('network/depth_image_input', Image, queue_size=10)

        # visualization
        self._setpoint_marker_pub = rospy.Publisher('network/marker_pose_setpoint', Marker, queue_size=10)
        self._goal_pose_marker_pub = rospy.Publisher('network/marker_goal', Marker, queue_size=10)

        # Services
        self._arming_client = rospy.ServiceProxy('mavros/cmd/arming',CommandBool) #mavros service for arming/disarming the robot
        self._set_mode_client = rospy.ServiceProxy('mavros/set_mode', SetMode) #mavros service for setting mode. Position commands are only available in mode OFFBOARD.
        self._change_param = rospy.ServiceProxy('/mavros/param/set', ParamSet)
        self.takeoffService = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        self._set_home_client = rospy.ServiceProxy('/mavros/cmd/set_home', CommandHome)

        # control timer
        self.debug_timer = rospy.Timer(rospy.Duration(1), self._debugCb)

        self.gazebo.pauseSim()

        # state
        self.episode_num = 0
        self.step_num = 0
        self.total_step = 0
        self.cumulated_episode_reward = 0
        self.last_obs = 0
        self.previous_distance_from_des_point = 0
        self.home_init = False

        # Variables for Subscribers callback
        self._current_state = State()
        self._current_pose = PoseStamped()
        self._current_gps = NavSatFix()
        self._depth_image_meter = 0
        self._pose_setpoint_avoidance = PoseStamped()
        self._local_odometry = Odometry()

        '''
        Goal settings
        '''
        self.goal_angle_noise_degree = 180  # random goal direction
        self.random_start_direction = True
        self.goal_distance = 60
        self._goal_pose = PoseStamped()

        '''
        Settings for control method
        '''
        self.action_num = 2         # 2 for 2d, 3 for 3d
        self.state_length = 4       # 2 for only position info, 4 for position and vel info
        self.control_method = 'vel' # acc or vel
        self.takeoff_hight = 5
        
        '''
        Settings for termination
        '''
        self.max_episode_step = 800
        self.accept_radius = 2
        self.accept_velocity_norm = 0.2 # For velocity control at goal position
        
        self.work_space_x_max = self.goal_distance + 10
        self.work_space_x_min = -self.work_space_x_max
        self.work_space_y_max = self.work_space_x_max
        self.work_space_y_min = -self.work_space_x_max
        self.work_space_z_max = 10
        self.work_space_z_min = 1
        self.max_vertical_difference = 5

        self.min_dist_to_obs_meters = 1  # min distance to obstacle

        '''
        observation space
        '''
        self.screen_height = 80
        self.screen_width = 100
        
        if self.obs_type == 'vector':
            self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(1), shape=(25+self.state_length,))
        else:
            self.observation_space = spaces.Box(low=np.float32(0), high=np.float32(255), shape=(self.screen_height, self.screen_width, 2), dtype=np.uint8)

        self._state_raw = np.zeros(self.state_length)
        self._state_norm = np.zeros(self.state_length)
        self.all_state_raw = np.zeros(6) # store all the 6 states

        '''
        action space
        '''
        # output forward vertital speed and yaw rate
        self.vel_xy_max = 3
        self.vel_xy_min = 0.5
        self.vel_z_max = 1
        self.vel_yaw_max = math.radians(30)

        if self.action_num == 3:
            self.action_space = spaces.Box(low=np.array([self.vel_xy_min, -self.vel_z_max, -self.vel_yaw_max]), \
                                            high=np.array([self.vel_xy_max, self.vel_z_max, self.vel_yaw_max]), \
                                            dtype=np.float32)
        elif self.action_num == 2:
            self.action_space = spaces.Box(low=np.array([self.vel_xy_min, -self.vel_yaw_max]), \
                                            high=np.array([self.vel_xy_max, self.vel_yaw_max]), \
                                            dtype=np.float32)
        else:
            print("error: action num must be 2 or 3")
        
        # for pretrain test
        # self.action_space = spaces.Box(low=np.array([-self.vel_yaw_max]), \
        #                                     high=np.array([self.vel_yaw_max]), \
        #                                     dtype=np.float32)

        rospy.logdebug('mavdrone_test_env_helei node initialized...')

# Methods for subscribers callback
    # ----------------------------
    def _poseCb(self, msg):
        self._current_pose = msg
        self.pose_local = msg

    def _local_vel_Cb(self, msg):
        self.vel_local = msg

    def _stateCb(self, msg):
        self._current_state = msg
    
    def _gpsCb(self, msg):
        self._current_gps = msg

    def _imageCb(self, msg):
        depth_image_msg = msg
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg)
        except CvBridgeError as e:
            print(e)

        (rows,cols) = cv_image.shape
        _depth_image_meter_ori = np.array(cv_image, dtype=np.float32)
        _depth_image_meter_ori[np.isnan(_depth_image_meter_ori)] = 18
        self._depth_image_meter = _depth_image_meter_ori
        # replace nan with max detection distance

    def _image_gazebo_Cb(self, msg):
        depth_image_msg = msg

        # transfer image from msg to cv2 image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding=depth_image_msg.encoding)
        except CvBridgeError as e:
            print(e)
        
        # get image in meters
        image = np.array(cv_image, dtype=np.float32)

        # deal with nan
        image[np.isnan(image)] = self.max_depth_meter_gazebo
        image_small = cv2.resize(image, (self.screen_width, self.screen_height), interpolation = cv2.INTER_AREA)
        self._depth_image_meter = np.copy(image_small)

        # get image gray (0-255)
        self._depth_image_meter = np.clip(self._depth_image_meter, 0.1, self.max_depth_meter_gazebo)
        # print('depth min:{:.2f} max{:.2f}'.format(self._depth_image_meter.min(), self._depth_image_meter.max()))
        image_gray = self._depth_image_meter / self.max_depth_meter_gazebo * 255
        image_gray_int = image_gray.astype(np.uint8)
        self._depth_image_gray = np.copy(image_gray_int)
        # cv2.imshow('depth gray', image_gray_int)
        # cv2.waitKey(1)
        
    def _pose_avoidanceCb(self, msg):
        self._pose_setpoint_avoidance = msg

    def _local_odomCb(self, msg):
        self._local_odometry = msg

    def _debugCb(self, event):
        pass

# Methods for OpenAI GYM environment
    # -------------------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.gazebo.unpauseSim()
        self.set_action_velocity(action)
        self.gazebo.pauseSim()

        obs = self._get_obs()
        done = self._is_done(obs)
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.too_close_to_obstacle(),
            'step_num': self.step_num
        }
        reward = self._compute_reward_new(obs, done, action)
        
        self.cumulated_episode_reward += reward
        self.step_num += 1
        self.total_step += 1

        # self.publish_train_info(self.last_obs, action, obs, done, reward, self.cumulated_episode_reward)
        self.last_obs = obs
    
        return obs, reward, done, info

    def reset(self):
        '''
        include reset sim and reset env
        reset_sim: reset simulation and ekf2
        reset_env: arm uav and takeoff 
        '''
        rospy.logdebug('reset function in')
        
        # reset world and ekf2
        self._reset_sim()

        # reset uav to takeoff position
        self._init_env_variables()

        self._update_episode()
        obs = self._get_obs()
        self.last_obs = obs
        self.gazebo.pauseSim()
        rospy.logdebug("reset function out")
        return obs

    def close(self):
        """
        Function executed when closing the environment.
        Use it for closing GUIS and other systems that need closing.
        :return:
        """
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")

# Methods for costum environment
    # -----------------------------------
    def _reset_sim(self):
        """
        Including ***ekf2 stop*** and ***ekf2 start*** routines in original function
        目前要解决的问题是：
        1. 无人机在空中无法disarm，尝试从固件入手
        2. 然后尝试先disarm，然后reset sim，最后reset ekf2.
        问题解决：在固件中设置使用强制arm and disarm

        遇到问题：
        stop ekf2之后出现断开连接，无法重新连接。。。。
        估计是需要旧版本固件。。。
        
        update 11-03:
        发现其实新版本固件是可以用的，但是在stop ekf之后，无法使用rospy.sleep()函数
        因为stop ekf之后，系统整个的仿真变得慢，所以每次stop之后直接start即可
        """
        rospy.logdebug("RESET SIM START --- DONT RESET CONTROLLERS")
        
        # 1. disarm
        self.gazebo.unpauseSim()
        rospy.logdebug('waiting for disarm...')
        self.DisarmForce()
        rospy.logdebug('disarm ok...')
        self.gazebo.pauseSim()

        # 2. reset sim 
        rospy.logdebug('reset sim...')
        self.gazebo.resetSim()
        rospy.logdebug('reset sim ok')

        # 3. check all system ready
        self.gazebo.unpauseSim()
        # self._check_all_systems_ready()
        self.gazebo.pauseSim()

        # 4. reset ekf2
        self.gazebo.unpauseSim()
        rospy.logdebug("reset ekf2 module")
        subprocess.Popen([self.px4_ekf2_path, "stop"])
        subprocess.Popen([self.px4_ekf2_path, "start"])
        subprocess.Popen([self.px4_ekf2_path, "status"])
        rospy.logdebug('wait for ekf ok')
        local_pose = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped)
        rospy.logdebug(local_pose)
        while math.sqrt(local_pose.pose.position.x * local_pose.pose.position.x + local_pose.pose.position.y * local_pose.pose.position.y) > 1:
            rospy.logdebug(local_pose.pose.position)
            local_pose = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped)
            rospy.sleep(0.5)
        
        rospy.logdebug('ekf2 reset over')
        
        rospy.logdebug('wait for arm...')
        self.Arm()
        
        rospy.logdebug('wait for takeoff...')
        self.TakeOff()
        
        self.gazebo.pauseSim()
        
        rospy.logdebug("RESET SIM END")

    def _init_env_variables(self):
        rospy.logdebug('_init_env_variables start')
        
        # reset goal position 
        self._change_goal_pose_random()
        goal_pose = self._goal_pose.pose.position
        rospy.loginfo("changed goal pose to {} {} {}".format(goal_pose.x, goal_pose.y, goal_pose.z))
        
        # For Info Purposes
        self.step_num = 0

        self.previous_distance_from_des_point = \
        self.get_distance_from_desired_point(self._current_pose.pose.position)

        rospy.logdebug('_init_env_variables end')

    def _update_episode(self):
        self.episode_num += 1
        self.cumulated_episode_reward = 0

    def _get_obs(self):
        if self.obs_type == 'vector':
            obs = self._get_obs_vector()
        else:
            obs = self._get_obs_image()
        
        return obs
    
    def _get_obs_image(self):
        # get depth image from current topic
        image = self._depth_image_gray.copy() # Note: check image format. Now is 0-black near 255-wight far

        # transfer image to image obs according to 0-far  255-nears
        image_obs = 255 - image

        # publish image_obs
        image_obs_msg = self.bridge.cv2_to_imgmsg(image_obs)
        self._depth_image_gray_input.publish(image_obs_msg)

        state_feature_array = np.zeros((self.screen_height, self.screen_width))

        state_feature = self._get_state_feature()

        state_feature_array[0, 0 : self.state_length] = state_feature

        image_with_state = np.array([image_obs, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)
        
        return image_with_state
    
    def _get_obs_vector(self):
        # get normalized obs for no_cnn
        image = self._depth_image_gray.copy() # Note: check image format. Now is 0-black near 255-wight far
        image_obs = 255 - image
        
        # publish image_obs
        image_obs_msg = self.bridge.cv2_to_imgmsg(image_obs)
        self._depth_image_gray_input.publish(image_obs_msg)
        
        # 将图像进行分割，然后得到最大值转换成feature
        split_row = 5
        split_col = 5
        
        v_split_list = np.vsplit(image_obs, split_col)
        
        split_final = []
        for i in range(split_col):
            h_split_list = np.hsplit(v_split_list[i], split_row)
            for j in range(split_row):
                split_final.append(h_split_list[j].max())
        
        img_feature = np.array(split_final) / 255.0
        
        state_feature = self._get_state_feature() / 255
        
        feature_all = np.concatenate((img_feature, state_feature), axis=0)
        # feature_all = state_feature[-1]
        # print(feature_all)
        
        return feature_all
            
    def _get_state_feature(self):
        '''
        get state feature with velocity!
        Airsim pose use NED SYSTEM
        Gazebo pose z-axis up is positive different from NED
        Gazebo twist using body frame
        Output： Normalized state feature (0~255) because the feature will input together with image
        '''
        goal_pose = self._goal_pose
        current_pose = self.pose_local
        current_vel = self.vel_local
        # get distance and angle in polar coordinate
        # transfer to 0~255 image formate for cnn
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        relative_pose_z = goal_pose.pose.position.z - current_pose.pose.position.z
        distance = math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2))
        relative_yaw = self._get_relative_yaw(current_pose, goal_pose)

        distance_norm = distance / self.goal_distance * 255
        vertical_distance_norm = (-relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255
        
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        current_vel_local = current_vel.twist
        linear_velocity_xy = current_vel_local.linear.x  # forward velocity
        linear_velocity_xy = math.sqrt(pow(current_vel_local.linear.x, 2) + pow(current_vel_local.linear.y, 2))
        linear_velocity_norm = linear_velocity_xy / self.vel_xy_max * 255
        linear_velocity_z = current_vel_local.linear.z  #  vertical velocity
        linear_velocity_z_norm = (linear_velocity_z / self.vel_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (-current_vel_local.angular.z / self.vel_yaw_max / 2 + 0.5) * 255  # TODO: check the sign of the 

        if self.state_length == 2:
            # 2d velocity control
            self.state_feature_raw = np.array([distance, relative_yaw])
            state_norm = np.array([distance_norm, relative_yaw_norm])
        elif self.state_length == 3:
            # 3d velocity control
            self.state_feature_raw = np.array([distance, relative_pose_z, relative_yaw])
            state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm])
        elif self.state_length == 4:
            # 2d acc control
            self.state_feature_raw = np.array([distance, relative_yaw, linear_velocity_xy, current_vel_local.angular.z])
            state_norm = np.array([distance_norm, relative_yaw_norm, linear_velocity_norm, angular_velocity_norm])
        else:
            # 3d acc control
            self.state_feature_raw = np.array([distance, relative_pose_z, relative_yaw, linear_velocity_xy, linear_velocity_z, current_vel_local.angular.z])
            state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])

        state_norm = np.clip(state_norm, 0, 255)
        self.state_feature_norm = state_norm / 255  # (0~1)

        self.all_state_raw = np.array([distance, relative_pose_z, relative_yaw, linear_velocity_xy, linear_velocity_z, current_vel_local.angular.z])

        return state_norm

    def get_expert_action(self):
        # get PX4 avoidance action as expert demo
        action = np.zeros(self.action_num)
        current_pose = self._current_pose
        desired_pose = self._pose_setpoint_avoidance
        
        x_err = desired_pose.pose.position.x - current_pose.pose.position.x
        y_err = desired_pose.pose.position.y - current_pose.pose.position.y
        z_err = desired_pose.pose.position.z - current_pose.pose.position.z
        
        forward_speed = math.sqrt(pow(x_err, 2) + pow(y_err, 2))
        if forward_speed > self.vel_xy_max:
            forward_speed = self.vel_xy_max
        if forward_speed < self.vel_xy_min:
            forward_speed = self.vel_xy_min
            
        explicit_quat1 = [desired_pose.pose.orientation.x, desired_pose.pose.orientation.y, desired_pose.pose.orientation.z, \
                            desired_pose.pose.orientation.w]
        explicit_quat2 = [current_pose.pose.orientation.x, current_pose.pose.orientation.y, \
                            current_pose.pose.orientation.z, current_pose.pose.orientation.w]
        
        yaw_setpoint = euler_from_quaternion(explicit_quat1)[2]
        yaw_current = euler_from_quaternion(explicit_quat2)[2]
        yaw_err = self._get_yaw_error(yaw_setpoint, yaw_current)
        
        # using only P for velocity control
        action[0] = forward_speed
        if self.action_num == 3:
            action[1] = z_err
        action[-1] = yaw_err * 2.8
        
        action[-1] = np.clip(action[-1], -self.vel_yaw_max, self.vel_yaw_max)
        
        return action
        
    def set_action_velocity(self, action):
        """
        Set linear and angular velocity according to action
        """
        velocity_cmd_body = PositionTarget()
        velocity_cmd_body.type_mask = 1479
        velocity_cmd_body.coordinate_frame = 8
        
        current_pose = self._current_pose

        velocity_cmd_body.velocity.x = action[0]
        velocity_cmd_body.velocity.y = 0
        if self.action_num == 2:
            z_error = self.takeoff_hight - current_pose.pose.position.z
            velocity_cmd_body.velocity.z = z_error
        else:
            velocity_cmd_body.velocity.z = action[1]
        velocity_cmd_body.yaw_rate = action[-1]
        
        self._action_velocity_body_pub.publish(velocity_cmd_body)
        self.publish_marker_velocity_pose(action)
        self.publish_marker_goal_pose(self._goal_pose)

        self._rate.sleep()
    
    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the drone
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        # get vel_setpoint
        if self.control_method == 'vel':
            # action is velocity setpoint
            vel_xy_sp = action[0]
            vel_z_sp = action[1]
            vel_yaw_sp = action[-1]
        elif self.control_method == 'acc':
            vel_xy_sp = self.all_state_raw[3] + action[0]
            vel_xy_sp = np.clip(vel_xy_sp, self.vel_xy_min, self.vel_xy_max)

            vel_z_sp = self.all_state_raw[4] + action[1]
            vel_z_sp = np.clip(vel_z_sp, -self.vel_z_max, self.vel_z_max)

            vel_yaw_sp = self.all_state_raw[5] + action[-1]
            vel_yaw_sp = np.clip(vel_yaw_sp, -self.vel_yaw_max, self.vel_yaw_max)

            self.action_vel = np.array([vel_xy_sp, vel_z_sp, vel_yaw_sp])

        # get pose setpoint
        current_yaw = self.get_current_yaw()
        yaw_speed = vel_yaw_sp
        yaw_setpoint = current_yaw + yaw_speed

        dx_body = vel_xy_sp
        dy_body = 0
        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)  

        pose_setpoint = PoseStamped()
        pose_setpoint.pose.position.x = self._current_pose.pose.position.x + dx_local
        pose_setpoint.pose.position.y = self._current_pose.pose.position.y + dy_local
        if self.action_num == 3:
            pose_setpoint.pose.position.z = self._current_pose.pose.position.z + vel_z_sp
        elif self.action_num == 2:
            pose_setpoint.pose.position.z = self.takeoff_hight

        orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]

        marker_network_setpoint = Marker()
        marker_network_setpoint.header.stamp = rospy.Time.now()
        marker_network_setpoint.header.frame_id = 'local_origin'
        marker_network_setpoint.type = Marker.SPHERE
        marker_network_setpoint.action = Marker.ADD
        marker_network_setpoint.pose.position = pose_setpoint.pose.position
        marker_network_setpoint.pose.orientation.x = 0.0
        marker_network_setpoint.pose.orientation.y = 0.0
        marker_network_setpoint.pose.orientation.z = 0.0
        marker_network_setpoint.pose.orientation.w = 1.0
        marker_network_setpoint.scale.x = 0.3
        marker_network_setpoint.scale.y = 0.3
        marker_network_setpoint.scale.z = 0.3
        marker_network_setpoint.color.a = 0.8
        marker_network_setpoint.color.r = 0.0
        marker_network_setpoint.color.g = 0.0
        marker_network_setpoint.color.b = 0.0

        self._local_pose_setpoint_pub.publish(pose_setpoint)
        self._setpoint_marker_pub.publish(marker_network_setpoint)

        # publish goal pose marker
        self.publish_marker_goal_pose(self._goal_pose)

        self._rate.sleep()
    
    def set_action_pose(self, action):
        '''
        set action using desired pose
        input: action_unscaled
        '''
        d_xy = action[0]
        if self.action_num == 3:
            d_z = action[1]
        d_yaw = action[-1]
        # transfer d_xy, d_z, d_yaw to pose data
        current_yaw = self.get_current_yaw()
        yaw_setpoint = current_yaw + d_yaw

        dx_body = d_xy
        dy_body = 0
        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

        pose_setpoint = PoseStamped()
        pose_setpoint.pose.position.x = self._current_pose.pose.position.x + dx_local
        pose_setpoint.pose.position.y = self._current_pose.pose.position.y + dy_local
        # pose_setpoint.pose.position.z = self._current_pose.pose.position.z + d_z
        if self.action_num == 3:
            pose_setpoint.pose.position.z = self._current_pose.pose.position.z + d_z
        elif self.action_num == 2:
            pose_setpoint.pose.position.z = self.takeoff_hight

        orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]

        marker_network_setpoint = Marker()
        marker_network_setpoint.header.stamp = rospy.Time.now()
        marker_network_setpoint.header.frame_id = 'local_origin'
        marker_network_setpoint.type = Marker.SPHERE
        marker_network_setpoint.action = Marker.ADD
        marker_network_setpoint.pose.position = pose_setpoint.pose.position
        marker_network_setpoint.pose.orientation.x = 0.0
        marker_network_setpoint.pose.orientation.y = 0.0
        marker_network_setpoint.pose.orientation.z = 0.0
        marker_network_setpoint.pose.orientation.w = 1.0
        marker_network_setpoint.scale.x = 0.3
        marker_network_setpoint.scale.y = 0.3
        marker_network_setpoint.scale.z = 0.3
        marker_network_setpoint.color.a = 0.8
        marker_network_setpoint.color.r = 0.0
        marker_network_setpoint.color.g = 0.0
        marker_network_setpoint.color.b = 0.0

        # publish pose setpoint and it's marker
        self._local_pose_setpoint_pub.publish(pose_setpoint)
        self._setpoint_marker_pub.publish(marker_network_setpoint)

        # publish goal pose marker
        self.publish_marker_goal_pose(self._goal_pose)

        self._rate.sleep()
        # TODO: calculate sleep time for every step

    def publish_marker_goal_pose(self, goal_pose):
        # publish goal pose marker
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'local_origin'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = goal_pose.pose.position
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.accept_radius * 2
        marker.scale.y = self.accept_radius * 2
        marker.scale.z = self.accept_radius * 2
        marker.color.a = 0.8
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        self._goal_pose_marker_pub.publish(marker)

    def publish_marker_velocity_pose(self, action):
        d_xy = action[0]
        d_z = action[1]
        d_yaw = action[-1]
        current_yaw = self.get_current_yaw()
        yaw_setpoint = current_yaw + d_yaw

        dx_body = d_xy
        dy_body = 0
        dx_local, dy_local = self.point_transfer(dx_body, dy_body, -yaw_setpoint)

        pose_setpoint = PoseStamped()
        pose_setpoint.pose.position.x = self._current_pose.pose.position.x + dx_local
        pose_setpoint.pose.position.y = self._current_pose.pose.position.y + dy_local
        if self.action_num == 3:
            pose_setpoint.pose.position.z = self._current_pose.pose.position.z + d_z
        elif self.action_num == 2:
            pose_setpoint.pose.position.z = self.takeoff_hight

        orientation_setpoint = quaternion_from_euler(0, 0, yaw_setpoint)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]

        marker_network_setpoint = Marker()
        marker_network_setpoint.header.stamp = rospy.Time.now()
        marker_network_setpoint.header.frame_id = 'local_origin'
        marker_network_setpoint.type = Marker.SPHERE
        marker_network_setpoint.action = Marker.ADD
        marker_network_setpoint.pose.position = pose_setpoint.pose.position
        marker_network_setpoint.pose.orientation.x = 0.0
        marker_network_setpoint.pose.orientation.y = 0.0
        marker_network_setpoint.pose.orientation.z = 0.0
        marker_network_setpoint.pose.orientation.w = 1.0
        marker_network_setpoint.scale.x = 0.3
        marker_network_setpoint.scale.y = 0.3
        marker_network_setpoint.scale.z = 0.3
        marker_network_setpoint.color.a = 0.8
        marker_network_setpoint.color.r = 0.0
        marker_network_setpoint.color.g = 0.0
        marker_network_setpoint.color.b = 0.0

        self._setpoint_marker_pub.publish(marker_network_setpoint)

    def get_avoidance_output_state_old(self):
        """
        transfer from avoidance msg: /mavros/setpoint_position/local_avoidance to
        scaled network output: forward speed and yaw rate in (-1, 1)
        """
        current_pose = self._current_pose
        desired_pose = self._pose_setpoint_avoidance

        dx_local = desired_pose.pose.position.x - current_pose.pose.position.x
        dy_local = desired_pose.pose.position.y - current_pose.pose.position.y
        dz_local = desired_pose.pose.position.z - current_pose.pose.position.z

        explicit_quat1 = [desired_pose.pose.orientation.x, desired_pose.pose.orientation.y, desired_pose.pose.orientation.z, \
                            desired_pose.pose.orientation.w]
        explicit_quat2 = [current_pose.pose.orientation.x, current_pose.pose.orientation.y, \
                            current_pose.pose.orientation.z, current_pose.pose.orientation.w]
        yaw_setpoint = euler_from_quaternion(explicit_quat1)[2]
        yaw_current = euler_from_quaternion(explicit_quat2)[2]

        # transfer dx dy from local frame to body frame
        dx_body, dy_body = self.point_transfer(dx_local, dy_local, yaw_current)
        yaw_speed = self._get_yaw_error(yaw_setpoint, yaw_current)

        # print(dx_local, dy_local, math.degrees(yaw_current), dx_body, dy_body, math.degrees(yaw_setpoint))

        if dx_body < 0:
            dx_body = 0
        if dx_body > self.vel_xy_max:
            dx_body = self.vel_xy_max

        avoidance_output = np.array([dx_body / self.vel_xy_max * 2 - 1, dz_local / self.vel_z_max, yaw_speed / self.vel_yaw_max])
        avoidance_output = np.clip(avoidance_output, -1, 1)

        return avoidance_output

    def get_avoidance_output_state(self):
        '''
        generate action from px4 avoidance output
        avoidance output: /mavros/setpoint_position/local_avoidance
        action_unscaled: d_xy, d_z, d_yaw
        '''
        self.gazebo.unpauseSim()
        # wait for msgs
        current_pose = rospy.wait_for_message('/mavros/setpoint_position/local_avoidance', PoseStamped)
        desired_pose = rospy.wait_for_message('mavros/local_position/pose', PoseStamped)

        dx_local = desired_pose.pose.position.x - current_pose.pose.position.x
        dy_local = desired_pose.pose.position.y - current_pose.pose.position.y
        d_xy = math.sqrt(dx_local*dx_local + dy_local*dy_local)
        d_z = desired_pose.pose.position.z - current_pose.pose.position.z

        explicit_quat1 = [desired_pose.pose.orientation.x, desired_pose.pose.orientation.y, desired_pose.pose.orientation.z, \
                            desired_pose.pose.orientation.w]
        explicit_quat2 = [current_pose.pose.orientation.x, current_pose.pose.orientation.y, \
                            current_pose.pose.orientation.z, current_pose.pose.orientation.w]
        yaw_setpoint = euler_from_quaternion(explicit_quat1)[2]
        yaw_current = euler_from_quaternion(explicit_quat2)[2]

        yaw_different = self._get_yaw_error(yaw_setpoint, yaw_current)

        print("yaw_sp: {:.2f} yaw: {:.2f} yaw_diff: {:.2f}".format(math.degrees(yaw_setpoint), math.degrees(yaw_current), math.degrees(yaw_different)))
        
        if d_xy < 0.1:
            d_yaw = yaw_different
        else:
            # d_yaw = math.atan(dx_local/dy_local)
            d_yaw = yaw_different
        
        yaw_diff_max_degree = 45
        
        if d_yaw > math.radians(yaw_diff_max_degree):
            d_yaw = math.radians(yaw_diff_max_degree)
        elif d_yaw < -math.radians(yaw_diff_max_degree):
            d_yaw = -math.radians(yaw_diff_max_degree)

        action = np.array([d_xy, -d_z, -d_yaw])

        print("yaw_sp: {:.2f} yaw: {:.2f} yaw_diff: {:.2f} d_yaw: {:.2f}". \
            format(math.degrees(yaw_setpoint), math.degrees(yaw_current), math.degrees(yaw_different), math.degrees(d_yaw)))

        self.gazebo.pauseSim()

        return action

    def point_transfer(self, x, y, theta):
        # transfer x, y to another frame
        x1 = x * math.cos(theta) + y * math.sin(theta)
        x2 = - x * math.sin(theta) + y * math.cos(theta)

        return x1, x2

    def get_avoidance_action_unscaled(self):
        current_pose = self._current_pose.pose.position
        desired_pose = self._pose_setpoint_avoidance.pose.position

        action_unscaled = np.array([desired_pose.x - current_pose.x, desired_pose.y - current_pose.y])
        action_unscaled = np.clip(action_unscaled, -self._speed_xy_max, self._speed_xy_max)

        return action_unscaled

    def _is_done(self, obs):
        """
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It detected something with the sonar that is too close
        3) It flipped due to a crash or something
        4) It has reached the desired point
        5) It is too close to the obstacle
        """

        episode_done = False
        current_pose = self._current_pose.pose
        current_position = current_pose.position
        current_orientation = current_pose.orientation

        is_inside_workspace_now = self.is_inside_workspace()
        has_reached_des_pose    = self.is_in_desired_pose()
        too_close_to_obstable   = self.too_close_to_obstacle()

        too_much_steps = (self.step_num > self.max_episode_step)

        rospy.logdebug(">>>>>> DONE RESULTS <<<<<")

        if not is_inside_workspace_now:
            rospy.loginfo("is_inside_workspace_now=" +
                         str(is_inside_workspace_now))
        if has_reached_des_pose:
            rospy.loginfo("has_reached_des_pose="+str(has_reached_des_pose))
        if too_close_to_obstable:
            rospy.loginfo("has crashed to the obstacle=" + str(too_close_to_obstable))
        if too_much_steps:
            rospy.loginfo("too much steps=" + str(too_much_steps))

        # We see if we are outside the Learning Space
        episode_done = not(is_inside_workspace_now) or\
                        has_reached_des_pose or\
                        too_close_to_obstable or\
                        too_much_steps
    
        return episode_done

    def _compute_reward(self, obs, done, action):
        '''
        get reward from action and obs
        input action is unscaled action
        '''
        reward = 0
        # get distance from goal position
        distance_now = self.get_distance_from_desired_point(self._current_pose.pose.position)

        if not done:
            reward_dist_old = self.previous_distance_from_des_point - distance_now
            self.previous_distance_from_des_point = distance_now

            # change reward according to the dist to goal
            dist_coeff = (1 - np.clip(distance_now / self.goal_distance, 0, 1)) * 3
            reward_dist =  dist_coeff * reward_dist_old
            # print('dist: {:.2f} r_old: {:.2f} dist_coeff: {:.2f} r_new: {:.2f}'.format(distance_now, reward_dist_old, dist_coeff, reward_dist))

            if self.action_num == 3:
                action_punishment_1 = 0.0 * (1 - action[0]/self.vel_xy_max)
                action_punishment_2 = 0.1 * abs(action[1] / self.vel_z_max)
                action_punishment_3 = 0.1 * abs(action[2] / self.vel_yaw_max)
                action_punishment_all = action_punishment_1 + action_punishment_2 + action_punishment_3
            else:
                action_punishment_1 = 0.0 * (1 - action[0]/self.vel_xy_max)
                action_punishment_3 = 0.1 * abs(action[-1] / self.vel_yaw_max)
                action_punishment_all = action_punishment_1 + action_punishment_3
            reward = reward_dist - action_punishment_all

            # print('reward: {:.2f} punishment: {:.2f} final: {:.2f}'.format(reward_dist, action_punishment_all, reward))

            # if self.control_method == 'vel':
            #     action_punishment = 0.1 * (abs(action[-1]/self.vel_yaw_max))
            # else:
            #     action_punishment = 0.1 * (abs(action[0]/self.acc_xy_max) + abs(action[-1]/self.acc_yaw_max))

            # reward -= action_punishment

        else:
            if self.is_in_desired_pose():
                reward = 10
            if self.too_close_to_obstacle():
                reward = -10

        # self._reward_pub.publish(reward)

        return reward

    def _compute_reward_new(self, obs, done, action):
        '''
        get reward from action and obs
        input action is unscaled action
        总结一下 ：
        1. 稀疏奖励：到达 10 碰撞 -10 出圈 -10
        2. 连续奖励：
            距离缩小 总70
        3. 惩罚：
            位置惩罚：离直线越近越好（分为水平和高度）
            动作惩罚：动作越小越好
            *时间惩罚：隐性，时间越长上面的惩罚越大
            障碍物惩罚：接近障碍物会有惩罚（尽量远离）
        '''
        reward = 0
        # get distance from goal position
        distance_now = self.get_distance_from_desired_point(self._current_pose.pose.position)

        if not done:
            # 计算连续奖励
            reward_dist_old = self.previous_distance_from_des_point - distance_now
            self.previous_distance_from_des_point = distance_now

            # change reward according to the dist to goal
            # dist_coeff = (1 - np.clip(distance_now / self.goal_distance, 0, 1)) * 3
            dist_coeff = 1
            reward_dist =  dist_coeff * reward_dist_old
            # reward_dist = math.exp(-distance_now*distance_now/50)

            # Position Punishment
            x = self._current_pose.pose.position.x
            y = self._current_pose.pose.position.y
            z = self._current_pose.pose.position.z
            x_g = self._goal_pose.pose.position.x
            y_g = self._goal_pose.pose.position.y
            z_g = self._goal_pose.pose.position.z
            punishment_xy = np.clip(self.getDis(x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy

            # Action punishment
            if self.action_num == 3:
                action_punishment_1 = 0.0 * (1 - action[0]/self.vel_xy_max)
                action_punishment_2 = 0.05 * abs(action[1] / self.vel_z_max)
                action_punishment_3 = 0.05 * abs(action[2] / self.vel_yaw_max)
                punishment_action = action_punishment_1 + action_punishment_2 + action_punishment_3
            else:
                action_punishment_1 = 0.0 * (1 - action[0]/self.vel_xy_max)
                action_punishment_2 = 0
                action_punishment_3 = abs(action[-1] / self.vel_yaw_max)
                punishment_action = action_punishment_1 + action_punishment_3

            # Obstacle punishment
            punishment_obs = 1 - np.clip((self._depth_image_meter.min() - self.min_dist_to_obs_meters) / 5, 0, 1)
            # print(punishment_obs)
            
            # action acc punishment
            punishment_action_acc = 0
            if self.state_length == 4:
                # 如果是2D 包含速度信息，可以使用action_acc punishment
                current_velocity_xy = self.state_feature_raw[2]
                current_yaw_rate = self.state_feature_raw[3]
                acc_xy = abs(action[0] - current_velocity_xy) / self.vel_xy_max / 2
                acc_yaw = abs(action[1] - current_yaw_rate) / self.vel_yaw_max / 2
                punishment_action_acc = acc_xy  + acc_yaw 
                
                # print(acc_xy, acc_yaw, punishment_action_acc)
             
            # 原始 reward
            # reward = reward_dist - 0.05 * punishment_action - 0.01 * punishment_pose - 0.01 * punishment_obs*punishment_obs
            # 新 reward 增加对于action突变引起的 punishment 让控制更顺滑
            reward = reward_dist - 0.1 * punishment_action - 0.1 * punishment_action_acc - 0.1 * punishment_pose - 0.1 * punishment_obs * punishment_obs
            # reward = reward * 10
            # reward = reward_dist - punishment_pose - punishment_action
            # reward = reward_dist - 0.01

            # print('r_dis: {:.2f} p_xy: {:.2f} p_z: {:.2f} p_a2: {:.2f} p_a3: {:.2f} r_all: {:.2f}'.format(
            #     reward_dist, punishment_xy, punishment_z, action_punishment_2, action_punishment_3, reward
            # ))

        else:
            if self.is_in_desired_pose():
                reward = 10
            if self.too_close_to_obstacle() or not self.is_inside_workspace():
                reward = -10

        return reward

    def update_goal_distance(self, goal_dist):
        self.goal_distance = goal_dist
        self.work_space_x_max = self.goal_distance + 10
        self.work_space_x_min = -self.work_space_x_max
        self.work_space_y_max = self.work_space_x_max
        self.work_space_y_min = -self.work_space_x_max
        print('goal dist is set to : ', self.goal_distance)

# Methods for PX4 control
    # --------------------------------------------
    def TakeOff(self):
        # method 2: using offboard mode to take off 
        pose_setpoint = PoseStamped()
        pose_setpoint.pose.position.x = 0
        pose_setpoint.pose.position.y = 0
        pose_setpoint.pose.position.z = self.takeoff_hight
        
        # random yaw angle at start point
        if self.random_start_direction:
            yaw_angle_rad = math.pi * (2 * random.random() - 1)
        else:
            yaw_angle_rad = 0
        orientation_setpoint = quaternion_from_euler(0, 0, yaw_angle_rad)
        pose_setpoint.pose.orientation.x = orientation_setpoint[0]
        pose_setpoint.pose.orientation.y = orientation_setpoint[1]
        pose_setpoint.pose.orientation.z = orientation_setpoint[2]
        pose_setpoint.pose.orientation.w = orientation_setpoint[3]
        for i in range(10):
            self._local_pose_setpoint_pub.publish(pose_setpoint)

        # self.setMavMode('OFFBOARD', 5)
        self._set_mode_client(0, 'OFFBOARD')

        while self._current_pose.pose.position.z < self.takeoff_hight - 0.2:
            # 1. publish topics
            self._local_pose_setpoint_pub.publish(pose_setpoint)

            # 2. check arm status, if not, arm first
            rospy.logdebug('Armed: {}, Mode： {}'.format(self._current_state.armed, self._current_state.mode))
            if not self._current_state.armed:
                rospy.logdebug('ARM AGAIN')
                self._arming_client.call(True)
                
            if self._current_state.mode != 'OFFBOARD':
                rospy.logdebug('SET OFFBOARD AGAIN')
                # self.setMavMode('OFFBOARD', 5)
                self._set_mode_client(0, 'OFFBOARD')
            
            rospy.sleep(0.1)

        rospy.logdebug("Took off success")
    
    def Arm(self):
        rospy.logdebug("wait for armed")
        rospy.wait_for_service("mavros/cmd/arming")

        while not self._current_state.armed:
            self._arming_client.call(True)
            rospy.logdebug('try to arm')
            rospy.sleep(0.5)

        rospy.logdebug("ARMED!!!")

    def Disarm(self):
        rospy.logdebug("wait for disarmed")
        rospy.wait_for_service("mavros/cmd/arming")

        while self._current_state.armed:
            self._arming_client.call(False)
            rospy.sleep(0.1)

        rospy.logdebug("DISARMED!!!")

    def DisarmForce(self):
        rospy.logdebug("wait for disarmed force")
        rospy.wait_for_service("/mavros/cmd/command")

        ret = self._mavros_cmd_client.call(False,
                                           400, 0,
                                           0.0, 21196.0,
                                           0.0, 0.0, 0.0, 0.0, 0.0)
        while not ret.success:
            ret = self._mavros_cmd_client.call(False,
                                               400, 0,
                                               0.0, 21196.0,
                                               0.0, 0.0, 0.0, 0.0, 0.0)
            rospy.sleep(0.1)
        rospy.logdebug("DISARMED!!!")

    def SetHomePose(self):
        '''
        this function is very slow, need to be optimized
        '''
        # set home position to current gps position
        # print(self._current_gps.latitude, self._current_gps.longitude)
        rospy.logdebug("Setting home pose...")

        ret_sethome = self._set_home_client(current_gps=True, latitude=self._current_gps.latitude, longitude=self._current_gps.longitude, \
                                                    altitude=self._current_gps.altitude)

        while not ret_sethome.success:
            rospy.logwarn("Failed to set home, try again")
            ret_sethome = self._set_home_client(current_gps=True, latitude=self._current_gps.latitude, longitude=self._current_gps.longitude, \
                                                    altitude=self._current_gps.altitude)
        
        self.home_init = True

        rospy.logdebug("Set home pose success!")

    def setMavMode(self, mode, timeout):
        """mode: PX4 mode string, timeout(int): seconds"""
        rospy.logdebug("setting FCU mode: {0}".format(mode))
        old_mode = self._current_state.mode
        loop_freq = 5  # Hz
        rate_new = rospy.Rate(loop_freq)
        mode_set = False
        for i in range(timeout * loop_freq):
            if self._current_state.mode == mode:
                mode_set = True
                rospy.logdebug("set mode success | seconds: {0} of {1}".format(i / loop_freq, timeout))
                break
            else:
                rospy.logdebug('current mode: {0}, try to set to {1}'.format(self._current_state.mode, mode))
                try:
                    res = self._set_mode_client(0, mode)  # 0 is custom mode
                    if not res.mode_sent:
                        rospy.logerr("failed to send mode command")
                except rospy.ServiceException as e:
                    rospy.logerr(e)
            rate_new.sleep()

# Some useful methods
    # --------------------------------------------
    def get_distance_from_desired_point(self, current_position):
        """
        Calculates the distance from the current position to the desired point
        :param start_point:
        :return:
        """
        curr_position = np.array([current_position.x, current_position.y, current_position.z])
        des_position = np.array([self._goal_pose.pose.position.x,\
                                self._goal_pose.pose.position.y,\
                                self._goal_pose.pose.position.z])
        distance = self.get_distance_between_points(curr_position, des_position)

        return distance

    def get_distance_between_points(self, p_start, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        a = np.array(p_start)
        b = np.array(p_end)

        distance = np.linalg.norm(a - b)

        return distance
    
    def getDis(self,pointX,pointY,lineX1,lineY1,lineX2,lineY2):
        '''
        Get distance between Point and Line
        Used to calculate position punishment
        '''
        a=lineY2-lineY1
        b=lineX1-lineX2
        c=lineX2*lineY1-lineX1*lineY2
        dis=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
        
        return dis

    def too_close_to_obstacle(self):
        """
        Check the distance to the obstacle using the depth perception
        """
        too_close = False

        if self._depth_image_meter.min() < self.min_dist_to_obs_meters:
            too_close = True

        return too_close

    def is_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = False
        current_position = self._current_pose.pose.position

        if current_position.x > self.work_space_x_min and current_position.x <= self.work_space_x_max:
            if current_position.y > self.work_space_y_min and current_position.y <= self.work_space_y_max:
                if current_position.z > self.work_space_z_min and current_position.z <= self.work_space_z_max:
                    is_inside = True

        return is_inside

    def is_in_desired_pose(self):
        """
        It return True if the current position is similar to the desired position
        """
        in_desired_pose = False
        current_pose = self._current_pose
        current_pose = self._current_pose
        current_velocity = self._local_odometry.twist.twist.linear
        velocity = np.array([current_velocity.x, current_velocity.y, current_velocity.z])
        # print(current_velocity, np.linalg.norm(velocity,ord=1))
        # if self.get_distance_from_desired_point(current_pose.pose.position) < self.accept_radius and \
        #      np.linalg.norm(velocity,ord=1) < self.accept_velocity_norm:
        if self.get_distance_from_desired_point(current_pose.pose.position) < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def _get_yaw_error(self, desired_yaw, curr_yaw):
        angular_error = desired_yaw - curr_yaw
        if angular_error > math.pi:
            angular_error -= math.pi * 2
        elif angular_error < -math.pi:
            angular_error += math.pi * 2
        
        return angular_error

    def get_current_yaw(self):
        orientation = self._current_pose.pose.orientation

        current_orientation = [orientation.x, orientation.y, \
                                orientation.z, orientation.w]
            
        current_attitude = euler_from_quaternion(current_orientation)
        current_yaw = euler_from_quaternion(current_orientation)[2]

        return current_yaw
    
    def _get_relative_yaw(self, current_pose, goal_pose):
        # get relative angle
        relative_pose_x = goal_pose.pose.position.x - current_pose.pose.position.x
        relative_pose_y = goal_pose.pose.position.y - current_pose.pose.position.y
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        explicit_quat = [current_pose.pose.orientation.x, current_pose.pose.orientation.y, \
                                current_pose.pose.orientation.z, current_pose.pose.orientation.w]
        
        yaw_current = euler_from_quaternion(explicit_quat)[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def get_current_pose(self):
        return np.array([self._current_pose.pose.position.x, self._current_pose.pose.position.y, self._current_pose.pose.position.z])
    
    def _change_goal_pose_random(self):
        distance = self.goal_distance

        noise = np.random.random() * 2 - 1
        angle = noise * math.radians(self.goal_angle_noise_degree)

        goal_x = distance * math.cos(angle)
        goal_y = distance * math.sin(angle)

        self._goal_pose = PoseStamped()
        self._goal_pose.pose.position.x = goal_x
        self._goal_pose.pose.position.y = goal_y
        self._goal_pose.pose.position.z = self.takeoff_hight

        self._goal_pose_pub.publish(self._goal_pose)

    def publish_train_info(self, last_obs, action, obs, done, reward, total_reward):
        info = TrainInfo()
        info.header.stamp = rospy.Time.now()

        # Train info
        info.episode = self.episode_num
        info.step = self.step_num
        info.total_step = self.total_step

        # Current state
        info.goal_position = self._goal_pose.pose.position
        info.current_position = self._current_pose.pose.position

        last_obs_feature = last_obs[..., -1]
        last_obs_feature = last_obs_feature[0, : self.state_length]
        
        # obs
        obs_feature = obs[..., -1]
        obs_feature = obs_feature[0, : self.state_length]

        info.last_obs = last_obs_feature
        info.curr_obs = obs_feature

        info.state_feature = self.all_state_raw
        info.state_feature_norm = self.state_feature_norm*2 - 1

        info.min_dist_to_obs = self._depth_image_meter.min()

        # action
        info.action_scaled = self.scale_action(self.action_space, action)
        info.action_unscaled = action

        # reward        
        info.reward = reward
        info.total_reward = total_reward

        # done
        info.done = done

        # model
        if self.model is not None:
            with th.no_grad():
                unscaled_action, _ = self.model.predict(last_obs, deterministic=False)  # 在需要得到feature的时候，要对model进行一次传播
                info.action_original = unscaled_action

                if self.use_ppo == True:
                    feature_all = self.model.policy.features_extractor.feature_all.cpu().numpy()[0]
                    info.feature_all = feature_all
                    
                    if self.policy == 'cnn_mobile':
                        gap_out = self.model.policy.features_extractor.gap_out.cpu().numpy()[0].reshape(-1)
                        info.gap_out = gap_out[0:20]
                    # if self.policy == 'no_cnn':
                    #     print('a')
                else:
                    # get feature all
                    feature_all = self.model.actor.features_extractor.feature_all.cpu().numpy()[0]
                    info.feature_all = feature_all

                    if self.policy == 'cnn_mobile':
                        gap_out = self.model.actor.features_extractor.gap_layer_out.cpu().numpy()[0].reshape(-1)
                        info.gap_out = gap_out[0:20]

                    # get q-value for td3
                    last_obs = last_obs.swapaxes(0, 1)
                    last_obs = last_obs.swapaxes(0, 2)
                    q_value_current = self.model.critic(th.from_numpy(last_obs[tuple([None])]).float().cuda(), th.from_numpy(action[None]).float().cuda())
                    q_1 = q_value_current[0].cpu().numpy()[0]
                    q_2 = q_value_current[1].cpu().numpy()[0]
                    info.q_value_current = np.array([q_1, q_2])
                    info.q_value_next = np.array([0, 0])

                    q_value = min(q_1, q_2)[0]

                    self.visual_log_q_value(q_value, action, reward)

                    if self.policy == 'cnn_gap':
                        self.visual_CNN_heat_maps(last_obs)

        self._train_info_pub.publish(info)

    def scale_action(self, action_space, action):
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action_space: (gym.spaces.box.Box)
        :param action: (np.ndarray)
        :return: (np.ndarray)
        """
        low, high = action_space.low, action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0  
    
    def visual_log_q_value(self, q_value, action, reward):
        '''
        Create grid map (map_size = work_space)
        Log Q value and the best action in grid map
        At any grid position, record:
        1. Q value
        2. action 0
        3. action 1
        4. steps
        5. reward
        Save image every 10k steps
        '''

        # create init array if not exist 
        map_size = self.work_space_x_max*2 + 1
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size, map_size), np.nan)
        
        # record info
        pose_x = self._current_pose.pose.position.x
        pose_y = self._current_pose.pose.position.y
        pose_z = self._current_pose.pose.position.z # used for 3D case

        index_x = int(np.round(pose_x) + self.work_space_x_max)
        index_y = int(np.round(pose_y) + self.work_space_x_max)

        # check if index valid
        if index_x in range(0, map_size) and index_y in range(0, map_size):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print('Error: X:{} and Y:{} is outside of range 0~mapsize (visual_log_q_value)')

        # save array every record_step steps 
        record_step = 5000
        if self.total_step % record_step == 0:
            np.save('q_value_map_{}'.format(self.total_step), self.q_value_map)
            # refresh 5 6 7 8 to record period data
            self.q_value_map[5, :, :] = np.nan
            self.q_value_map[6, :, :] = np.nan
            self.q_value_map[7, :, :] = np.nan
            self.q_value_map[8, :, :] = np.nan

    def visual_CNN_heat_maps(self, obs_used):
        '''
        Generate CNN heat maps for every layers
        Output all weights and bias
        Output gradient
        '''

        step = self.total_step
        obs = obs_used
        
        layer_1_out = self.model.actor.features_extractor.layer_1_out.cpu().numpy()[0]      # (8, 40, 50)
        layer_2_out = self.model.actor.features_extractor.layer_2_out.cpu().numpy()[0]      # (8, 20, 25)
        layer_3_out = self.model.actor.features_extractor.layer_3_out.cpu().numpy()[0]      # (8, 10, 12)
        gap_layer_out = self.model.actor.features_extractor.gap_layer_out.cpu().numpy()[0]  # (8, 1, 1)

        msg = FeatureMaps()
        msg.header.stamp = rospy.Time.now()

        msg.obs = obs.reshape(-1)    # (2, 80, 100)

        msg.layer_1_out = layer_1_out.reshape(-1)
        msg.layer_2_out = layer_2_out.reshape(-1)
        msg.layer_3_out = layer_3_out.reshape(-1)
        msg.gap_layer_out = gap_layer_out.reshape(-1)

        msg.w_1 = self.model.actor.features_extractor.conv1[0].weight.cpu().numpy().reshape(-1)         # (8, 1, 3, 3)
        msg.b_1 = self.model.actor.features_extractor.conv1[0].bias.cpu().numpy().reshape(-1)
        msg.w_2 = self.model.actor.features_extractor.conv2[0].weight.cpu().numpy().reshape(-1)         # (8, 1, 3, 3)
        msg.b_2 = self.model.actor.features_extractor.conv2[0].bias.cpu().numpy().reshape(-1)
        msg.w_3 = self.model.actor.features_extractor.conv3[0].weight.cpu().numpy().reshape(-1)         # (8, 1, 3, 3)
        msg.b_3 = self.model.actor.features_extractor.conv3[0].bias.cpu().numpy().reshape(-1)

        self.feature_maps_pub.publish(msg)
