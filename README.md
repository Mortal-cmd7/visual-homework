# visual——homework
该代码实现功能如下：
1.自动识别对象并框选（每个对象一个独特的id）
2.鼠标点击人后显示其人体骨架以及人体运动轨迹
3.鼠标点击对象后显示对象距离摄像头的距离（需要开始前按“k”键来进行人为标定）

运行指令如下：
命令行参数说明
参数	                               说明	                  默认值
--source	                           视频源：                0 表示摄像头，也可以是视频文件路径或 RTSP 流	"0"
--view-img / --no-view-img	         是否显示实时窗口	      默认显示
--save-video	                       是否保存输出视频	      默认不保存
--output	                           输出视频保存路径      	"tracking_output.avi"
--conf	                             置信度阈值	            0.3
--iou	                               IoU 阈值	              0.3
--device	                           推理设备              	"cpu"
--no-fps	                           隐藏 FPS 显示	          默认显示
--show-conf	                         在标签上显示置信度分数	  默认不显示
--track-len	                         轨迹历史最大长度（帧数）	30
