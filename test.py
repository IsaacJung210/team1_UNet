from net.Unet import Unet
import torch, rospy, cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np

from message_filters import TimeSynchronizer, Subscriber
from torchvision import transforms as transforms


# PyTorch 모델 초기화
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Available Device = {device}")
model = Unet(class_num_=1, depth_=4, image_ch_=3, target_ch_=6).to(device)
# data = torch.load('/home/isaac/carla-ros-bridge/pth2ros/unet_epoch022.pth')
data = torch.load('/home/junhyung/VSCODE_WS/Unet_team1/carla_pth/unet_epoch002.pth')
model.load_state_dict(data['model'])

# ROS 노드 초기화
rospy.init_node('pytorch_node')
cv_bridge = CvBridge()


def callback(img, img_info, dep, dep_info):
    process_image(img, img_info, dep, dep_info)


# 이미지를 처리하고 publish하는 함수 정의
def process_image(img_msg, img_info, dep, dep_info):
    # ROS 이미지 메시지를 OpenCV 이미지로 변환
    img = cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
    b,g,r,_ = cv2.split(img)
    img = torch.Tensor(np.array([[b,g,r]])/255.0).to(device)

    with torch.no_grad():
        model.eval()
        output = model(img)
        result = np.zeros((output.shape[2], output.shape[3], 3), dtype=np.uint8)
        result[(output[0,0,:,:] >= 1.0).cpu()] = np.array([255,255,255])

    processed_img_msg = cv_bridge.cv2_to_imgmsg(result, encoding="passthrough")
    processed_img_msg.header = img_msg.header
    processed_img_msg.header.frame_id = 'map'
    processed_img_msg.header.stamp = img_msg.header.stamp 
    # stamp는 원래 image msg의 타임스탬프 그대로 하는게 좋을 듯! 문법 맞는지는 체크해봐야됨!

    processed_img_pub.publish(processed_img_msg)
    processed_img_camera_info_pub.publish(img_info)
    processed_dep_pub.publish(dep)
    processed_dep_camera_info_pub.publish(dep_info)
    print("publish!")


sub_img = Subscriber('/carla/ego_vehicle/rgb_front/image', Image)
sub_img_info = Subscriber('/carla/ego_vehicle/rgb_front/camera_info', CameraInfo)
sub_dep = Subscriber('/carla/ego_vehicle/depth_front/image', Image)
sub_dep_info = Subscriber('/carla/ego_vehicle/depth_front/camera_info', CameraInfo)

ts = TimeSynchronizer([sub_img, sub_img_info, sub_dep, sub_dep_info], queue_size=1)
ts.registerCallback(callback)

processed_img_pub = rospy.Publisher('/processed_image', Image, queue_size=3)
processed_img_camera_info_pub = rospy.Publisher('/processed_image/camera_info', CameraInfo, queue_size=3)
processed_dep_pub = rospy.Publisher('/carla/ego_vehicle/depth_front/image', Image, queue_size=3)
processed_dep_camera_info_pub = rospy.Publisher('/carla/ego_vehicle/depth_front/camera_info', CameraInfo, queue_size=3)

# ROS 실행
rospy.spin()