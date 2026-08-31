import time
import numpy as np
import serial
import cv2

SERIAL_PORT = 'COM4'
CAMERA_INDEX = 1
RATE = 9600
LOWER_COLOR = np.array([35, 50, 50]) 
UPPER_COLOR = np.array([85, 255, 255])



try:
    # Try to connect to Arduino (optional)
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, RATE, timeout=1)
        time.sleep(2)
        print(f"Arduino Connection in {SERIAL_PORT} has been made")
    except serial.SerialException as e:
        print(f"Arduino not connected: {e}")
        print("Continuing without Arduino connection...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise IOError("Camera not found")
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"A {frame_width} x {frame_height} Camera has been found.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_area = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_area) > 100:
                M = cv2.moments(largest_area)
                if M["m00"] != 0:
                    cp_x = int(M["m10"] / M["m00"])
                    cp_y = int(M["m01"] / M["m00"])

                    offset_x = cp_x - (frame_width/2)
                    offset_y = -(cp_y - (frame_height * 0.8))

                    cur_pos = f"{offset_x}, {offset_y}\n"
                    if ser is not None:
                        ser.write(cur_pos.encode('utf-8'))

                    cv2.circle(frame, (cp_x, cp_y), 10, (0, 255, 0), -1)
                    cv2.putText(frame, f"X and Y points are {offset_x}, {offset_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow("Trash Can Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



except serial.SerialException as e:
    print(f"Serial Port has an error: {e}")
except Exception as e:
    print(f"Error occured: {e}")
finally:
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'ser' in locals() and ser is not None and ser.is_open:
        ser.close()
        print("Arduino Connection has been successfully closed")
    


                
                


    
    

