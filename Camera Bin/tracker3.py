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

# --- PID & MOTION SETTINGS ---
TARGET_LANDING_RADIUS = 100 

# Speed Settings
MAX_SPEED = 200    
MIN_SPEED = 80     

# Tuning Knobs
Kp_TURN = 0.4      
Kp_SPEED = 2.0     

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

            # --- NO ROTATION APPLIED ---
            # Using the original camera orientation

            # Resize for consistent math
            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]
            center_x = width // 2

            # Visuals
            cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

            # Image Processing
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            left_motor = 0
            right_motor = 0
            status = "SEARCHING"

            if len(contours) > 0:
                # Find largest blob
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                if radius > 10: 
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    
                    # --- PID LOGIC ---
                    
                    # 1. Steering Error
                    error_x = x - center_x 
                    
                    # 2. Speed Logic (Throttle)
                    dist_error = TARGET_LANDING_RADIUS - radius
                    
                    if dist_error < 0:
                        # Too Close -> Stop
                        forward_speed = 0
                        status = "LANDED"
                    else:
                        # Far Away -> Drive
                        speed_factor = dist_error / TARGET_LANDING_RADIUS 
                        forward_speed = MIN_SPEED + (speed_factor * (MAX_SPEED - MIN_SPEED))
                        forward_speed = int(np.clip(forward_speed, 0, MAX_SPEED))
                        status = "TRACKING"

                    # 3. Mixing
                    turn_correction = int(error_x * Kp_TURN)
                    
                    if status == "TRACKING":
                        # Differential Drive
                        left_motor = forward_speed + turn_correction
                        right_motor = forward_speed - turn_correction
                        
                        # Clamp
                        left_motor = max(min(left_motor, 255), -255)
                        right_motor = max(min(right_motor, 255), -255)
                    else:
                        left_motor = 0
                        right_motor = 0

            else:
                # NO OBJECT -> TANK TURN (SPIN)
                status = "SEARCHING (SPIN)"
                left_motor = -150  
                right_motor = 150

            # Display Status
            cv2.putText(frame, f"Mode: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"L: {left_motor} R: {right_motor}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Send Command
            cmd_str = f"M,{int(left_motor)},{int(right_motor)}\n"
            try:
                await client.write_gatt_char(CHARACTERISTIC_UUID, cmd_str.encode())
            except Exception:
                pass

            cv2.imshow("PID Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop on exit
        await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(run_tracker())