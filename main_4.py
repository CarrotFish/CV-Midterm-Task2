import cv2
from ultralytics import YOLO
import os, time

window_name = "Tracking & Counting"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow(window_name, 960, 540)

model = YOLO(os.path.abspath('./best.pt'))
cap = cv2.VideoCapture(os.path.abspath('./tracksource.mp4'))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (frame_width, frame_height)

# 判定线
line_st = (0.5, 0.0)
line_ed = (0.5, 1.0)

def getFramePos(line_pos: tuple[float])->tuple[int]:
    return (int(line_pos[0]*frame_size[0]), int(line_pos[1]*frame_size[1]))

LINE_ST = getFramePos(line_st)
LINE_ED = getFramePos(line_ed)

def getLineFuncValue(detect_pos: tuple[float])->float:
    return (LINE_ED[1]-LINE_ST[1])*detect_pos[0]-(detect_pos[1]-LINE_ST[1])*LINE_ED[0]+(detect_pos[1]-LINE_ED[1])*LINE_ST[0]

# 记忆字典：存放格式为 {Tracking_ID: 上一帧的 Y 坐标}
track_history = {}  

total_cross_count = 0  
# 记录已经计算过越线的 ID，防止一辆车在越线时因为边缘抖动被重复计数
counted_ids = set()    

running = True

while running:
    key = cv2.waitKey(1) & 0xFF
    # 按下q停止
    if key == ord('q'):
        break
    # 按下空格暂停
    elif key == 0x20: 
        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == 0x20:
                break
            if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                running = False
                break
            pass
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        running = False
        break
    if not cap.isOpened():
        time.sleep(0.01)
        continue
    success, frame = cap.read()
    if not success:
        time.sleep(0.01)
        continue
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", show=False)
    cv2.line(frame, LINE_ST, LINE_ED, (0, 255, 255), 5)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        for box, track_id in zip(boxes, track_ids):
            # 获取检测框中心坐标
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            # 位置判定
            if track_id in track_history:
                prev = track_history[track_id] # 调取该 ID 上一帧的坐标
                if getLineFuncValue(prev)*getLineFuncValue((cx, cy)) <= 0:
                    # 确保这个 ID 没被重复算过
                    if track_id not in counted_ids:
                        total_cross_count += 1
                        counted_ids.add(track_id) # 标记为已计数
                        print(f"检测到越线！Tracking ID: {track_id}，总数: {total_cross_count}")

            # 更新当前帧坐标到字典，为下一帧做准备
            track_history[track_id] = (cx, cy)

    # 显示结果
    cv2.putText(frame, f"Total Crossed: {total_cross_count}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv2.imshow(window_name, frame)
cap.release()
cv2.destroyAllWindows()