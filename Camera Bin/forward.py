import cv2
import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner

# --- CONFIGURATION ---
DEVICE_UUID = "033F2D12-9FF3-419C-95CA-5F746FA68B8E"
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
CAMERA_INDEX = 1

# Green Object Color
LOWER_COLOR = np.array([40, 50, 50])   
UPPER_COLOR = np.array([80, 255, 255])

# --- MOTION SETTINGS ---
TARGET_RADIUS = 100    
RADIUS_DEADBAND = 30   
MAX_SPEED = 150    
Kp_DIST = 1.5          

# --- MANUAL CENTER COORDINATES ---
CENTER_X = 127
CENTER_Y = 222
BOX_SIZE = 60 

async def run_tracker():
    print("Scanning for Robot...")
    device = await BleakScanner.find_device_by_address(DEVICE_UUID, timeout=10.0)
    if not device:
        print("Robot not found.")
        return

    print(f"Connecting to {device.name}...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    async with BleakClient(device) as client:
        print("Connected! Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            new_dimensions = (255, 445) # (width, height) expected by cv2.resize()

            # 1. Assign the returned value (the new image) to a single variable
            resized_frame = cv2.resize(frame, new_dimensions)

            # 2. Extract the height and width from the new image's shape attribute *after* resizing
            height, width = resized_frame.shape[:2]

            # --- DEFINE BOX ---
            box_left   = CENTER_X - BOX_SIZE
            box_right  = CENTER_X + BOX_SIZE
            box_top    = CENTER_Y - BOX_SIZE
            box_bottom = CENTER_Y + BOX_SIZE
            
            # Draw Crosshair (Blue) at (285, 222)
            cv2.line(frame, (CENTER_X, 0), (CENTER_X, height), (255, 0, 0), 2)
            cv2.line(frame, (0, CENTER_Y), (width, CENTER_Y), (255, 0, 0), 2)

            # Image Processing
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motor_speed = 0
            status = "WAITING"
            box_color = (255, 0, 255) # Default Purple (Not Locked)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                if radius > 10: 
                    # Visuals
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    
                    # --- CHECK LOCK CONDITIONS ---
                    is_in_box = (box_left < x < box_right) and (box_top < y < box_bottom)
                    is_dist_good = abs(TARGET_RADIUS - radius) < RADIUS_DEADBAND

                    # --- LOGIC PRIORITY ---
                    
                    if is_in_box and is_dist_good:
                        # 1. FULL LOCK -> FORCE STOP
                        motor_speed = 0
                        status = "LOCKED ON"
                        box_color = (0, 255, 0) # Green
                        
                    else:
                        # 2. NOT LOCKED -> DECIDE MOVEMENT
                        
                        if not is_dist_good:
                            # Problem: Distance is wrong (Too Far or Too Close)
                            dist_error = TARGET_RADIUS - radius
                            
                            pid_output = dist_error * Kp_DIST
                            if pid_output > 0:
                                motor_speed = pid_output + 60 
                                status = "APPROACHING"
                            elif pid_output < 0:
                                motor_speed = 0 
                                status = "TOO CLOSE"
                        
                        else:
                            # Problem: Distance is Good, but NOT in Box
                            # Behavior: Stop and wait for alignment
                            motor_speed = 0
                            status = "ALIGN ME!"

                    # Clamp Speed
                    motor_speed = max(min(motor_speed, MAX_SPEED), -MAX_SPEED)

            else:
                status = "LOST - STOP"
                motor_speed = 0

            # Draw the Box (Green if Locked, Purple otherwise)
            cv2.rectangle(frame, (box_left, box_top), (box_right, box_bottom), box_color, 2)

            # Status Overlay
            cv2.putText(frame, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Speed: {int(motor_speed)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Send Command
            cmd_str = f"M,{int(motor_speed)},{int(motor_speed)}\n"
            try:
                await client.write_gatt_char(CHARACTERISTIC_UUID, cmd_str.encode())
            except Exception:
                pass

            cv2.imshow("Strict Lock Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(run_tracker())