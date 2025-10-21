import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def pointcloud_callback(msg):
    # Print the header information
    rospy.loginfo("Header: %s", msg.header)

    # Get the fields of the PointCloud2 message
    fields = msg.fields
    rospy.loginfo("Fields:")
    for field in fields:
        rospy.loginfo("  Name: %s, Offset: %d, Data Type: %d, Count: %d", field.name, field.offset, field.datatype, field.count)

def listener():
    rospy.init_node('pointcloud_listener', anonymous=True)
    rospy.Subscriber("/semantic_pointcloud", PointCloud2, pointcloud_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()


# stamp: 
#   secs: 1681989369
#   nsecs:  64018011
# frame_id: "camera_color_optical_frame"
# [INFO] [1729573309.699614, 1681989369.139925]: Fields:
# [INFO] [1729573309.700248, 1681989369.139925]:   Name: x, Offset: 0, Data Type: 7, Count: 1
# [INFO] [1729573309.700889, 1681989369.139925]:   Name: y, Offset: 4, Data Type: 7, Count: 1
# [INFO] [1729573309.701517, 1681989369.139925]:   Name: z, Offset: 8, Data Type: 7, Count: 1
# [INFO] [1729573309.702146, 1681989369.139925]:   Name: rgb, Offset: 16, Data Type: 7, Count: 1
# [INFO] [1729573310.151102, 1681989369.361610]: Header: seq: 42



# stamp: 
#   secs: 1681989400
#   nsecs: 820454836
# frame_id: "camera_color_optical_frame"
# [INFO] [1729573373.220327, 1681989400.897900]: Fields:
# [INFO] [1729573373.220916, 1681989400.897900]:   Name: x, Offset: 0, Data Type: 7, Count: 1
# [INFO] [1729573373.221535, 1681989400.897900]:   Name: y, Offset: 4, Data Type: 7, Count: 1
# [INFO] [1729573373.222251, 1681989400.902938]:   Name: z, Offset: 8, Data Type: 7, Count: 1
# [INFO] [1729573373.222835, 1681989400.902938]:   Name: r, Offset: 12, Data Type: 2, Count: 1
# [INFO] [1729573373.223413, 1681989400.902938]:   Name: g, Offset: 13, Data Type: 2, Count: 1
# [INFO] [1729573373.224387, 1681989400.902938]:   Name: b, Offset: 14, Data Type: 2, Count: 1

# stamp: 
#   secs: 1681989368
#   nsecs: 130008459
# frame_id: "camera_color_optical_frame"
# [INFO] [1729573487.647197, 1681989368.231867]: Fields:
# [INFO] [1729573487.647953, 1681989368.231867]:   Name: x, Offset: 0, Data Type: 7, Count: 1
# [INFO] [1729573487.648670, 1681989368.231867]:   Name: y, Offset: 4, Data Type: 7, Count: 1
# [INFO] [1729573487.649256, 1681989368.231867]:   Name: z, Offset: 8, Data Type: 7, Count: 1
# [INFO] [1729573487.649851, 1681989368.231867]:   Name: r, Offset: 12, Data Type: 2, Count: 1
# [INFO] [1729573487.650382, 1681989368.231867]:   Name: g, Offset: 13, Data Type: 2, Count: 1
# [INFO] [1729573487.650932, 1681989368.231867]:   Name: b, Offset: 14, Data Type: 2, Count: 1
# [INFO] [1729573488.109160, 1681989368.459114]: Header: seq: 39