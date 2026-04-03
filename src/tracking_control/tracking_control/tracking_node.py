import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################
def cap_length(v, l):
    norm = np.norm(v)
    if norm <= l:
        return v
    return (v / norm) * l


def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obs_pose_callback(self, msg):
        self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        self.get_logger().info('Received Detected Goal Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.goal_pose = cp_world
        
    def get_current_poses(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        robot_pose = None
        obstacle_pose = None
        goal_pose = None
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_pose = np.array([robot_world_x, robot_world_y, robot_world_z])
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            
            if self.obs_pose != None:
                obstacle_pose = robot_world_R @ self.obs_pose + robot_pose

            if self.goal_pose != None:
                goal_pose = robot_world_R @ self.goal_pose + robot_pose
                
        except ValueError as e:
            self.get_logger().error('Value error: ' + str(e))

        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))

        except Exception as e:
            self.get_logger().error('Error: ' + str(e))
        
        return robot_pose, obstacle_pose, goal_pose
    
    start_pose = None
    def timer_update(self):
        ################### Write your code here ###################
        
        # Now, the robot stops if the object is not detected
        # But, you may want to think about what to do in this case
        # and update the command velocity accordingly
        if self.goal_pose is None:
            # Spin in place until the goal is seen
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 2.0
            self.pub_control_cmd.publish(cmd_vel)
            return
        
        # Get the current object pose in the robot base_footprint frame
        robot_pose, current_obs_pose, current_goal_pose = self.get_current_poses()
        
        if start_pose == None and robot_pose != None:
            start_pose = robot_pose

        # Get the control velocity command
        cmd_vel = self.controller(robot_pose, current_obs_pose, current_goal_pose)
        
        # publish the control command
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    reached_goal = False
    def controller(self, robot_pose, current_obs_pose, current_goal_pose):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        # If the robot or goal position is unknown, don't move
        if robot_pose == None or current_goal_pose == None:
            return Twist()

        # Constants
        BUFFER_DISTANCE = 0.2
        K_V1 = 0.1
        K_V2 = 0.2
        MAX_V = 5

        # Decide which pose to move towards
        if reached_goal
            target_pose = start_pose
        else
            target_pose = current_goal_pose

        # Determine distance and direction to target
        goal_diff = robot_pose - current_goal_pose
        goal_d = np.norm(goal_diff)
        goal_dir = goal_diff / goal_d

        # If at target
        if d <= BUFFER_DISTANCE:
            if not reached_goal:
                reached_goal = True
            return Twist()
                
        # Determine straight line velocity
        v = d * K_V1 * goal_dir

        # If there is an object, direct away from it
        if current_obs_pose != None:
            obj_diff = robot_pose - current_obj_pose
            obj_d = np.norm(obj_diff)
            obj_dir = obj_diff / obj_d

            # Get perpendicular vector to movement
            perp1 = np.array([-goal_dir[1], goal_dir[0], goal_dir[2]])
            perp2 = np.array([goal_dir[1], -goal_dir[0], goal_dir[2]])
            if np.dot(perp1, obj_dir) > 0:
                perp = perp1
            else:
                perp = perp2

            # Update velocity to move away from object
            s_mod = (1 / obj_d) * K_V2
            v_mod = s_mod * perp
            if obj_d <= BUFFER_DISTANCE:
                v = v_mod
            else:
                v = v + v_mod

        v = cap_length(v, MAX_V)
        cmd_vel = Twist()
        cmd_vel.linear.x = v[0]
        cmd_vel.linear.y = v[1]
        cmd_vel.linear.z = v[2]
        cmd_vel.angular.z = 0.0
        return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
