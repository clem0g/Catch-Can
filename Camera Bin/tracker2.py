import cv2
import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner

# --- BLUETOOTH CONFIGURATION ---
DEVICE_UUID = "033F2D12-9FF3-419C-95CA-5F746FA68B8E"
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# --- CAMERA & VISION CONFIGURATION ---
CAMERA_INDEX = 1
LOWER_COLOR = np.array([35, 50, 50]) 
UPPER_COLOR = np.array([85, 255, 255])

# --- LOGIC SETTINGS (TUNED) ---
TARGET_CENTER_X = 177 

# 1. WIDENED DEADBAND
# Was 20. Now 60.
# This means if the object is anywhere in the middle 30% of the screen, 
# the car will just drive forward instead of jittering left/right.
DEADBAND = 60          

# 2. EARLIER STOPPING
# Was 100. Now 130.
# This ensures the robot stops while the object is still fully in view, 
# rather than waiting until it crashes into the camera.
STOPPING_RADIUS = 130  

async def run_tracker():
    print(f"Scanning for HM-10...")
    device = await BleakScanner.find_device_by_address(DEVICE_UUID, timeout=10.0)
    
    if not device:
        print(f"Device Not Found.")
        return

    print(f"Connecting to {device.name}...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Camera.")
        return

    async with BleakClient(device) as client:
        print("Connected! Start Chasing...")
        last_command = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Standardize size
            frame = cv2.resize(frame, (640, 480))

            # Processing
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # DEFAULT COMMAND IS STOP
            # If we see nothing, we stop immediately.
            command = 'S' 

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                # Draw Visuals
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                
                # Draw the "Safe Zone" lines so you can see them
                left_boundary = TARGET_CENTER_X - DEADBAND
                right_boundary = TARGET_CENTER_X + DEADBAND
                cv2.line(frame, (left_boundary, 0), (left_boundary, 480), (255, 0, 0), 2)
                cv2.line(frame, (right_boundary, 0), (right_boundary, 480), (255, 0, 0), 2)

                # --- NEW LOGIC FLOW ---
                
                # 1. Is it close enough? -> STOP
                if radius > STOPPING_RADIUS:
                    command = 'S'
                    cv2.putText(frame, "LANDED - STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 2. Is it outside the boundaries? -> TURN ONLY (Don't move forward)
                elif x < left_boundary:
                    command = 'L'
                    cv2.putText(frame, "TURNING LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                elif x > right_boundary:
                    command = 'R'
                    cv2.putText(frame, "TURNING RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # 3. If we are here, we are centered and far away -> DRIVE
                else:
                    command = 'F' 
                    cv2.putText(frame, "FORWARD", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                cv2.putText(frame, "OBJECT LOST - STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- SEND COMMAND ---
            # We send 'S' repeatedly if lost to ensure it stops
            if command != last_command or command == 'S':
                try:
                    await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
                    last_command = command
                except Exception as e:
                    pass # Ignore random BLE hiccups

            cv2.imshow("Tracker", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        await client.write_gatt_char(CHARACTERISTIC_UUID, b'S')
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(run_tracker())