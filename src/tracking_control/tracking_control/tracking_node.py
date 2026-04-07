import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math


# Functions for quaternion and rotation matrix conversion
def hat(k):
    khat = np.zeros((3, 3))
    khat[0, 1] = -k[2]
    khat[0, 2] =  k[1]
    khat[1, 0] =  k[2]
    khat[1, 2] = -k[0]
    khat[2, 0] = -k[1]
    khat[2, 1] =  k[0]
    return khat


def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2 * q[0] * qhat + 2 * qhat2


def cap_length(v, l):
    norm = np.linalg.norm(v)
    if norm <= l or norm < 1e-9:
        return v
    return (v / norm) * l


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')

        # Current poses stored in world frame (odom)
        self.start_global_pos = None
        self.obs_pose = None
        self.goal_pose = None

        self.reached_goal = False
        self.returned_home = False
        self.avoid_side = 0.0   # +1 = left, -1 = right

        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)

        # Subscribers
        self.sub_detected_obs_pose = self.create_subscription(
            PoseStamped,
            'detected_color_object_pose',
            self.detected_obs_pose_callback,
            10
        )
        self.sub_detected_goal_pose = self.create_subscription(
            PoseStamped,
            'detected_color_goal_pose',
            self.detected_goal_pose_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(0.01, self.timer_update)

    def detected_obs_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # TODO: Filtering
        # Keep only reasonable detections in front of robot
        if (
            center_points[0] < 0.05 or
            center_points[0] > 3.0 or
            abs(center_points[1]) > 1.5 or
            center_points[2] < -0.2 or
            center_points[2] > 0.7
        ):
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y,
                transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y,
                transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error(f'Transform Error: {e}')
            return

        # Low-pass filter to reduce jitter
        alpha = 0.35
        if self.obs_pose is None:
            self.obs_pose = cp_world
        else:
            self.obs_pose = alpha * cp_world + (1.0 - alpha) * self.obs_pose

    def detected_goal_pose_callback(self, msg):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

        # TODO: Filtering
        if (
            center_points[0] < 0.05 or
            center_points[0] > 3.0 or
            abs(center_points[1]) > 1.5 or
            center_points[2] < -0.2 or
            center_points[2] > 0.7
        ):
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x,
                transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x,
                transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return

        alpha = 0.35
        if self.goal_pose is None:
            self.goal_pose = cp_world
        else:
            self.goal_pose = alpha * cp_world + (1.0 - alpha) * self.goal_pose

    def get_current_poses(self):
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        robot_pose = None
        robot_world_R = None

        try:
            # transform of base_footprint expressed in odom
            transform = self.tf_buffer.lookup_transform(
                odom_id,
                'base_footprint',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )

            robot_pose = np.array([transform.transform.translation.x, transform.transform.translation.y,
                transform.transform.translation.z])

            robot_world_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x,
                transform.transform.rotation.y, transform.transform.rotation.z]))

        except TransformException as e:
            self.get_logger().error(f'Transform error: {e}')
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

        # obs_pose and goal_pose are already stored in world frame
        return robot_pose, robot_world_R, self.obs_pose, self.goal_pose

    def timer_update(self):
        ################### Write your code here ###################
        robot_global_pose, robot_world_R, current_obs_pose, current_goal_pose = self.get_current_poses()

        if robot_global_pose is None or robot_world_R is None:
            self.pub_control_cmd.publish(Twist())
            return

        if self.start_global_pos is None:
            self.start_global_pos = robot_global_pose.copy()

        # if goal has not been seen yet in this instance, spin to look for it (doesn't require repeated searching)
        if current_goal_pose is None and not self.reached_goal:
            cmd_vel = Twist()
            cmd_vel.angular.z = 0.20
            self.pub_control_cmd.publish(cmd_vel)
            return

        cmd_vel = self.controller(
            robot_global_pose,
            robot_world_R,
            current_obs_pose,
            current_goal_pose
        )

        self.pub_control_cmd.publish(cmd_vel)

    def controller(self, current_robot_global_pos, robot_world_R, current_obs_pose, current_goal_pose):
        # If the robot or goal position is unknown, don't move
        if current_robot_global_pos is None or robot_world_R is None:
            return Twist()

        # breaking apart goal reach and returned home
        if self.returned_home:
            return Twist()

        # Choosing target, continuing seperation of phases
        if self.reached_goal:
            if self.start_global_pos is None:
                return Twist()
            target_world = self.start_global_pos
        else:
            if current_goal_pose is None:
                return Twist()
            target_world = current_goal_pose

        # cacluating target vectors in/to robot frame robot_world_R.T @ target_vec_world
        target_vec_world = target_world - current_robot_global_pos
        target_vec_robot = robot_world_R.T @ target_vec_world
        target_planar = target_vec_robot[:2]
        target_dist = np.linalg.norm(target_planar)

        TARGET_DISTANCE = 0.30
        GOAL_K = 0.8
        MAX_LINEAR = 0.12
        MAX_ANGULAR = 0.45

        # Stop at goal, then return to start. Stop again at home.
        if target_dist < TARGET_DISTANCE:
            self.avoid_side = 0.0
            if not self.reached_goal:
                self.reached_goal = True
                return Twist()
            else:
                self.returned_home = True
                return Twist()

        # attarction to target
        v_planar = GOAL_K * target_planar
        v_planar = cap_length(v_planar, MAX_LINEAR)

        # robot frame obstacle for avoidance
        obstacle_active = False
        if current_obs_pose is not None:
            obs_vec_world = current_obs_pose - current_robot_global_pos
            obs_vec_robot = robot_world_R.T @ obs_vec_world
            obs_x = obs_vec_robot[0]
            obs_y = obs_vec_robot[1]
            obs_dist = np.linalg.norm(obs_vec_robot[:2])

            # only avoiding when obstacle is actually close to robot?? values are estimated
            if (obs_x > -0.05) and (obs_x < 0.80) and (abs(obs_y) < 0.45) and (obs_dist < 0.90):
                obstacle_active = True

                # making sure obs passing side is stuck to
                if self.avoid_side == 0.0:
                    if abs(target_planar[1]) > 0.08:
                        self.avoid_side = np.sign(target_planar[1])
                    elif abs(obs_y) > 0.05:
                        self.avoid_side = -np.sign(obs_y)
                    else:
                        self.avoid_side = 1.0

                # repulsion from obstacle center
                away = -obs_vec_robot[:2]
                away_norm = np.linalg.norm(away)
                if away_norm > 1e-6:
                    away = away / away_norm
                else:
                    away = np.array([0.0, 0.0])

                repulse_mag = max(0.0, 0.18 * (1.0 / max(obs_dist, 0.18) - 1.0 / 0.90))
                slide_mag = max(0.0, 0.12 * (1.0 - obs_dist / 0.90))

                v_planar = v_planar + repulse_mag * away + np.array([0.0, slide_mag * self.avoid_side])

                # velocity killing when it's close to obs
                if obs_dist < 0.35:
                    v_planar[0] = min(v_planar[0], 0.05)

                v_planar = cap_length(v_planar, MAX_LINEAR)

            # remove obstacle effect once it is passed (clearing variables in case robot turns around
            # and sees "new" obstacle). google said elif supposedly is good to use when multiple alternatives 
            # are involved??
            elif (obs_x < -0.10) or (obs_dist > 1.10):
                self.avoid_side = 0.0

        # heading control kinematic model, hopefully stops when passing obstacle 
        heading_error = math.atan2(target_planar[1], max(target_planar[0], 1e-3))
        if obstacle_active:
            w = 0.0
        else:
            # should point back to target (goal or home depending on phase) after passing obstacle
            w = float(np.clip(1.0 * heading_error, -MAX_ANGULAR, MAX_ANGULAR))

        cmd_vel = Twist()
        cmd_vel.linear.x = float(v_planar[0])
        cmd_vel.linear.y = float(v_planar[1])
        cmd_vel.linear.z = 0.0
        cmd_vel.angular.z = w
        return cmd_vel


def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
