import cv2
import numpy as np
import time
import asyncio
import math
from bleak import BleakClient, BleakScanner

# ==========================================
# --- CONFIGURATION & CONSTANTS ---
# ==========================================

DEVICE_UUID = "033F2D12-9FF3-419C-95CA-5F746FA68B8E"
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

CAMERA_INDEX = 1        
FRAME_WIDTH = 255       
FRAME_HEIGHT = 490
CENTER_X = 127
FLOOR_Y = 400           

LOWER_COLOR = np.array([40, 50, 50])   
UPPER_COLOR = np.array([80, 255, 255])

FOCAL_LENGTH = 364.2    
REAL_RADIUS_CM = 6.85   
DEPTH_OFFSET_CM = -10   
GRAVITY_REAL = 980      

BLE_SEND_RATE = 0.1     
MAX_SPEED = 200         

# --- MOTION CONTROL SETTINGS ---
Kp = 2.5   
Kd = 1.0   
BASE_FORWARD_SPEED = 200

# !!! CALIBRATION REQUIRED !!!
# How many cm does your robot move in 1 second at BASE_FORWARD_SPEED?
# Increase this if it stops TOO EARLY. Decrease if it stops TOO LATE.
ROBOT_SPEED_CM_S = 100.0  

# Distance to stop before the target (so we don't run over the ball)
STOP_DISTANCE_CM = 0.0

# ==========================================
# --- PROJECTILE PREDICTOR CLASS ---
# ==========================================
class ProjectilePredictor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_x = None
        self.last_z = None
        self.last_y = None
        self.last_time = None
        self.smooth_z = None
        self.status = "IDLE"
        self.landing_history = []  
        self.locked_landing = None 

    def update(self, x_cm, raw_z_cm, y_px):
        # 1. --- DEAD ZONE CHECKS ---
        if x_cm is None: 
            return None
        
        now = time.time()
        
        # 2. --- Z-SMOOTHING ---
        if self.smooth_z is None: 
            self.smooth_z = raw_z_cm
        else: 
            self.smooth_z = (self.smooth_z * 0.7) + (raw_z_cm * 0.3)

        z_cm = self.smooth_z # This is now your HEIGHT
        
        
        # 3. --- COORDINATE TRANSFORMATION (Pixels -> Real CM) ---
        # We do this BEFORE the reset check so we can save the real Y value.
        
        # Invert Y logic: Top of screen (Negative offset) = Forward (Positive Distance)
        y_center = 230 # Assuming 480 height
        y_offset = y_px - y_center 
        y_cm = -1 * (y_offset * z_cm) / FOCAL_LENGTH 


        # 4. --- TIMEOUT / RESET ---
        # If we lost tracking for too long, reset the memory
        if self.last_time is None or (now - self.last_time > 0.5):
            self.reset()
            # CRITICAL CHANGE: We save y_cm (Real Distance), not y_px (Pixels)
            self.last_x, self.last_y, self.last_z, self.last_time = x_cm, y_cm, z_cm, now
            self.smooth_z = z_cm
            self.status = "TRACKING..."
            return None

        dt = now - self.last_time
        if dt == 0: 
            return self.locked_landing
            
        # If we already locked a target, keep returning it
        if self.locked_landing:
            return self.locked_landing


        # 5. --- VELOCITY & PHYSICS (Upward Camera Logic) ---
        vx_cm = (x_cm - self.last_x) / dt
        vy_cm = (y_cm - self.last_y) / dt # Forward Speed
        vz_cm = (z_cm - self.last_z) / dt # Vertical Speed (Gravity)

        # Solve for when Z (Height) hits 0
        a = 0.5 * GRAVITY_REAL 
        b = -vz_cm 
        c = -z_cm   

        delta = b**2 - 4*a*c

        if delta >= 0:
            # Quadratic Formula
            sqrt_delta = math.sqrt(delta)
            t1 = (-b - sqrt_delta) / (2*a)
            t2 = (-b + sqrt_delta) / (2*a)
            t_impact = max(t1, t2)
            
            if t_impact > 0:
                # Predict Landing Spot on Floor (Linear Motion)
                pred_x = x_cm + (vx_cm * t_impact)
                pred_y = y_cm + (vy_cm * t_impact) # Forward Distance
                
                # We only care if the ball is landing in a valid spot
                self.landing_history.append((pred_x, pred_y))
                
                # AVERAGE & LOCK
                if len(self.landing_history) >= 5:
                    avg_x = sum([p[0] for p in self.landing_history]) / len(self.landing_history)
                    avg_y = sum([p[1] for p in self.landing_history]) / len(self.landing_history)
                    
                    self.locked_landing = (avg_x, avg_y)
                    self.status = f"LOCKED: {avg_y:.0f}cm" # Display Forward Distance
                else:
                    self.status = f"GATHERING... {len(self.landing_history)}/5"
            else:
                self.status = "TRACKING (Past Camera)"
        else:
            self.status = "TRACKING (No Impact)"
        
        # Update Memory
        self.last_x, self.last_y, self.last_z, self.last_time = x_cm, y_cm, z_cm, now
        return self.locked_landing
predictor = ProjectilePredictor()

# ==========================================
# --- MAIN ASYNC LOOP ---
# ==========================================
async def run_tracker():
    print("--- SEARCHING FOR ROBOT ---")
    device = await BleakScanner.find_device_by_address(DEVICE_UUID, timeout=10.0)
    if not device:
        print(f"Error: Device not found.")
        return

    print(f"Connecting to {device.name}...")
    
    async with BleakClient(device) as client:
        print("Connected! Starting...")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        last_ble_time = 0 
        prev_error_x = 0
        
        # DISTANCE TRACKING VARIABLES
        remaining_distance = None
        was_locked = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Visuals
                cv2.line(frame, (20, 0), (20, FRAME_HEIGHT), (0, 0, 255), 1)
                cv2.line(frame, (FRAME_WIDTH-15, 0), (FRAME_WIDTH-15, FRAME_HEIGHT), (0, 0, 255), 1)

                # --- 1. IMAGE PROCESSING ---
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=3) 
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    
                    if radius > 2 and 20 < x < (FRAME_WIDTH - 15):
                        raw_dist = (REAL_RADIUS_CM * FOCAL_LENGTH) / radius
                        dist_cm = raw_dist + DEPTH_OFFSET_CM 
                        pixel_offset_x = CENTER_X - x
                        x_pos_cm = (pixel_offset_x * dist_cm) / FOCAL_LENGTH
                        
                        predictor.update(x_pos_cm, dist_cm, y)
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

                # --- 2. MOTION CONTROL ---
                landing = predictor.locked_landing
                now = time.time()
                dt = now - last_ble_time
                
                # A. Handle Lock State / Reset Distance
                if landing and not was_locked:
                    # New throw detected! Initialize distance
                    remaining_distance = landing[1] 
                    was_locked = True
                    print(f"TARGET LOCKED: {remaining_distance:.1f}cm away")
                elif not landing:
                    # Ball lost or timeout
                    was_locked = False
                    remaining_distance = None

                if dt > BLE_SEND_RATE:
                    if landing and remaining_distance is not None:
                        pred_x = landing[0] # Steering target
                        
                        # --- PID STEERING ---
                        error_x = pred_x
                        d_error = (error_x - prev_error_x)
                        turn_output = (Kp * error_x) + (Kd * d_error)
                        prev_error_x = error_x
                        if abs(error_x) < 5: turn_output = 0

                        # --- FORWARD CONTROL & INTEGRATION ---
                        forward_speed = 0
                        
                        # Only drive if:
                        # 1. We are roughly facing the target (error < 25cm)
                        # 2. We haven't reached the stop distance yet
                        if abs(error_x) < 25 and remaining_distance > STOP_DISTANCE_CM:
                            forward_speed = BASE_FORWARD_SPEED
                            
                            # ESTIMATE DISTANCE TRAVELED
                            # Distance = Speed * Time
                            # We multiply by BLE_SEND_RATE (approx dt)
                            cm_moved = ROBOT_SPEED_CM_S * dt
                            remaining_distance -= cm_moved
                        
                        # If we arrived, stop motors
                        if remaining_distance <= STOP_DISTANCE_CM:
                            forward_speed = 0
                            turn_output = 0
                            status_text = "ARRIVED"
                        else:
                            status_text = f"DRIVING... {remaining_distance:.0f}cm"

                        # Mix Motors
                        left_motor = int(forward_speed + turn_output)
                        right_motor = int(forward_speed - turn_output)
                        
                        left_motor = max(-MAX_SPEED, min(MAX_SPEED, left_motor))
                        right_motor = max(-MAX_SPEED, min(MAX_SPEED, right_motor))
                        
                        command = f"M,{left_motor},{right_motor}\n"
                        await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
                        
                    else:
                        await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
                        prev_error_x = 0
                        status_text = predictor.status
                    
                    last_ble_time = now

                    # --- VISUALS ---
                    cv2.putText(frame, status_text, (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
                    
                    # Map
                    map_size = 100
                    map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                    map_cx = map_size // 2
                    scale = 0.5
                    cv2.circle(map_img, (map_cx, map_size-5), 4, (255, 0, 0), -1)
                    
                    if landing and remaining_distance is not None:
                        # Draw target relative to current estimated distance
                        mz = int(map_size - (remaining_distance * scale))
                        mx = int(map_cx + (landing[0] * scale))
                        if 0 <= mx < map_size and 0 <= mz < map_size:
                            cv2.line(map_img, (mx-3, mz-3), (mx+3, mz+3), (0, 0, 255), 1)
                            cv2.line(map_img, (mx+3, mz-3), (mx-3, mz+3), (0, 0, 255), 1)

                    frame[10:10+map_size, FRAME_WIDTH-10-map_size:FRAME_WIDTH-10] = map_img
                    cv2.rectangle(frame, (FRAME_WIDTH-10-map_size, 10), (FRAME_WIDTH-10, 10+map_size), (255, 255, 255), 1)
                    cv2.imshow("Final Catcher", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            print("Stopping...")
            await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(run_tracker())
    except KeyboardInterrupt:
        print("Interrupted.")