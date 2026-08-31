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

# --- CALIBRATION CONSTANTS ---
# Calculated from your data (49.9px @ 50cm with 6.85cm radius)
FOCAL_LENGTH = 364.2 
REAL_RADIUS_CM = 6.85  

# Robot Center Alignment (From your previous test)
CENTER_X = 127
CENTER_Y = 222

async def run_tracker():
    print("Scanning for Robot...")
    device = await BleakScanner.find_device_by_address(DEVICE_UUID, timeout=10.0)
    if not device:
        print("Robot not found.")
        return

    print(f"Connecting to {device.name}...")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    async with BleakClient(device) as client:
        print("Connected! Flight Recorder Mode (No Motion).")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            height, width = frame.shape[:2]

            # 1. Draw the Reference Center (Blue Crosshair)
            cv2.line(frame, (CENTER_X, 0), (CENTER_X, height), (255, 0, 0), 1)
            cv2.line(frame, (0, CENTER_Y), (width, CENTER_Y), (255, 0, 0), 1)

            # Image Processing
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                if radius > 5: 
                    # --- 3D MATH SECTOR ---
                    
                    # 1. Calculate DEPTH (Z)
                    # Formula: Distance = (Real_Radius * Focal_Length) / Pixel_Radius
                    dist_cm = (REAL_RADIUS_CM * FOCAL_LENGTH) / radius
                    
                    # 2. Calculate LATERAL POS (X)
                    # Formula: X = (Pixel_Offset_From_Center * Distance) / Focal_Length
                    pixel_offset_x = x - CENTER_X
                    x_pos_cm = (pixel_offset_x * dist_cm) / FOCAL_LENGTH

                    # --- VISUALIZATION ---
                    
                    # Draw Object
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                    # Draw 3D Data Text next to object
                    text_z = f"Z: {dist_cm:.1f} cm"
                    text_x = f"X: {x_pos_cm:.1f} cm"
                    
                    cv2.putText(frame, text_z, (int(x)+10, int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, text_x, (int(x)+10, int(y)+25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw Top-Down Map (Mini-map in corner)
                    # This helps you visualize what the robot thinks is happening
                    map_w, map_h = 200, 200
                    map_img = np.zeros((map_h, map_w, 3), dtype=np.uint8)
                    
                    # Scale factor for map: 1 pixel = 1 cm
                    map_cx = map_w // 2
                    map_obj_x = int(map_cx + x_pos_cm)
                    map_obj_z = int(map_h - dist_cm) # Bottom is 0 distance
                    
                    # Draw Robot at bottom center of map
                    cv2.circle(map_img, (map_cx, map_h-10), 5, (255, 0, 0), -1) 
                    
                    # Draw Ball on map (Clip to fit)
                    if 0 <= map_obj_x < map_w and 0 <= map_obj_z < map_h:
                        cv2.circle(map_img, (map_obj_x, map_obj_z), 5, (0, 255, 255), -1)
                        cv2.line(map_img, (map_cx, map_h-10), (map_obj_x, map_obj_z), (100, 100, 100), 1)

                    # Overlay map on main frame
                    frame[10:210, 10:210] = map_img
                    cv2.rectangle(frame, (10, 10), (210, 210), (255, 255, 255), 1)
                    cv2.putText(frame, "Top-Down View", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("3D Coordinate Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop motors just in case
        await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(run_tracker())