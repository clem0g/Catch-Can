import cv2
import numpy as np
import time
import asyncio
import math
from bleak import BleakClient, BleakScanner

# ==========================================
# --- CONFIGURATION & CONSTANTS ---
# ==========================================

# BLE UUIDs (Replace with your specific device UUID if it changes)
DEVICE_UUID = "033F2D12-9FF3-419C-95CA-5F746FA68B8E"
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# CAMERA SETTINGS
CAMERA_INDEX = 1        # 0 for internal webcam, 1 for external USB
FRAME_WIDTH = 255       # Low resolution for speed
FRAME_HEIGHT = 490
CENTER_X = 127
FLOOR_Y = 400           # The Y-pixel coordinate considered the "ground"

# COLOR DETECTION (Green Object)
# Expanded range to account for motion blur
LOWER_COLOR = np.array([30, 40, 40])   
UPPER_COLOR = np.array([90, 255, 255])

# CALIBRATION
FOCAL_LENGTH = 364.2    # Calculated from previous tests
REAL_RADIUS_CM = 6.85   # Radius of the ball in cm
DEPTH_OFFSET_CM = -10   # Adjust based on camera mounting position

# PHYSICS
GRAVITY_REAL = 980      # Gravity in cm/s^2

# CONTROL SETTINGS
BLE_SEND_RATE = 0.1     # 10Hz (Send command every 0.1s)
TURN_GAIN = 3.0         # Proportional Gain (Higher = faster turning)
MAX_SPEED = 200         # Max PWM (0-255)

# ==========================================
# --- PROJECTILE PREDICTOR CLASS ---
# ==========================================
class ProjectilePredictor:
    def __init__(self):
        self.last_x = None
        self.last_z = None
        self.last_y = None
        self.last_time = None
        self.smooth_z = None
        self.predicted_landing = None
        self.status = "IDLE"

    def reset(self):
        """Resets the tracking state (e.g., when ball is lost)."""
        self.last_x = None
        self.last_z = None
        self.last_y = None
        self.last_time = None
        self.smooth_z = None
        self.predicted_landing = None
        self.status = "IDLE"

    def update(self, x_cm, raw_z_cm, y_px):
        now = time.time()
        
        # 1. Z-SMOOTHING (Low-pass filter to reduce jitter in depth)
        if self.smooth_z is None:
            self.smooth_z = raw_z_cm
        else:
            self.smooth_z = (self.smooth_z * 0.7) + (raw_z_cm * 0.3)
        z_cm = self.smooth_z

        # Check for timeout or new throw (reset if gap > 0.5s)
        if self.last_time is None or (now - self.last_time > 0.5):
            self.reset()
            self.smooth_z = z_cm
            self.last_x, self.last_z, self.last_y, self.last_time = x_cm, z_cm, y_px, now
            self.status = "TRACKING..."
            return None

        dt = now - self.last_time
        if dt == 0: return self.predicted_landing

        # 2. CALCULATE VELOCITY
        vy_px = (y_px - self.last_y) / dt
        vx_cm = (x_cm - self.last_x) / dt
        vz_cm = (z_cm - self.last_z) / dt

        # 3. DYNAMIC GRAVITY SCALING
        # Gravity acts on pixels differently depending on depth (Z)
        if z_cm < 1: z_cm = 1 
        g_px = (GRAVITY_REAL * FOCAL_LENGTH) / z_cm 
        
        # 4. SOLVE QUADRATIC EQUATION for Impact Time
        # 0.5*g*t^2 + vy*t + (y_current - y_floor) = 0
        a = 0.5 * g_px
        b = vy_px
        c = y_px - FLOOR_Y 

        delta = b**2 - 4*a*c
        
        if delta >= 0:
            # Calculate both roots, take the positive one
            t1 = (-b + math.sqrt(delta)) / (2*a)
            t2 = (-b - math.sqrt(delta)) / (2*a)
            t_impact = max(t1, t2)
            
            if t_impact > 0:
                # 5. PREDICT LANDING COORDINATES
                pred_x = x_cm + (vx_cm * t_impact)
                pred_z = z_cm + (vz_cm * t_impact)
                
                self.predicted_landing = (pred_x, pred_z)
                self.status = f"LOCKED: X{pred_x:.0f} Z{pred_z:.0f}"
            else:
                self.status = "TRACKING (Past Floor)"
        else:
            self.status = "TRACKING (No Impact)"
        
        # Update history for next frame
        self.last_x, self.last_z, self.last_y, self.last_time = x_cm, z_cm, y_px, now
        return self.predicted_landing

# Instantiate the predictor
predictor = ProjectilePredictor()

# ==========================================
# --- MAIN ASYNC LOOP ---
# ==========================================
async def run_tracker():
    print("--- SEARCHING FOR ROBOT ---")
    device = await BleakScanner.find_device_by_address(DEVICE_UUID, timeout=10.0)
    if not device:
        print(f"Error: Device {DEVICE_UUID} not found.")
        return

    print(f"Connecting to {device.name}...")
    
    async with BleakClient(device) as client:
        print("Connected! Starting Video Stream...")
        
        cap = cv2.VideoCapture(CAMERA_INDEX)
        # Force low resolution for higher FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        last_ble_time = 0 
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret: break

                # Safe Zone Visuals (Red lines on sides)
                safe_margin = 25 
                cv2.line(frame, (safe_margin, 0), (safe_margin, FRAME_HEIGHT), (0, 0, 255), 1)
                cv2.line(frame, (FRAME_WIDTH-safe_margin, 0), (FRAME_WIDTH-safe_margin, FRAME_HEIGHT), (0, 0, 255), 1)

                # --- 1. IMAGE PROCESSING ---
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
                mask = cv2.erode(mask, None, iterations=1)
                mask = cv2.dilate(mask, None, iterations=3) 
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                curr_x_cm = 0
                curr_z_cm = 0
                is_safe = False
                
                if len(contours) > 0:
                    c = max(contours, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    
                    # Ignore small noise and objects on the extreme edges
                    if radius > 2 and safe_margin < x < (FRAME_WIDTH - safe_margin):
                        is_safe = True
                        
                        # Calculate Depth (Z) and Horizontal Pos (X)
                        raw_dist = (REAL_RADIUS_CM * FOCAL_LENGTH) / radius
                        dist_cm = raw_dist + DEPTH_OFFSET_CM 
                        
                        pixel_offset_x = CENTER_X - x
                        x_pos_cm = (pixel_offset_x * dist_cm) / FOCAL_LENGTH
                        
                        curr_x_cm = x_pos_cm
                        curr_z_cm = dist_cm

                        # UPDATE PREDICTOR
                        predictor.update(x_pos_cm, dist_cm, y)
                        
                        # Draw visual feedback
                        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.putText(frame, f"{int(dist_cm)}cm", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    else:
                        cv2.putText(frame, "EDGE/NOISE", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                # --- 2. MOTION CONTROL STRATEGY ---
                landing = predictor.predicted_landing
                now = time.time()
                
                if now - last_ble_time > BLE_SEND_RATE:
                    if landing:
                        pred_x, pred_z = landing
                        
                        # CONTROL LOGIC:
                        # If pred_x is Positive -> Ball lands to the LEFT of camera center (in this coord system)
                        # Depending on motor wiring, you might need to swap these signs.
                        
                        error_x = pred_x
                        turn_cmd = int(error_x * TURN_GAIN)
                        
                        # Clamp speed
                        turn_cmd = max(-MAX_SPEED, min(MAX_SPEED, turn_cmd))
                        
                        # Differential Steering
                        # If turn_cmd > 0 (Left), Left Motor Back / Right Motor Fwd
                        left_motor = turn_cmd
                        right_motor = -turn_cmd
                        
                        command = f"M,{left_motor},{right_motor}\n"
                        # print(f"CMD: {command.strip()} | Tgt: {pred_x:.1f}") 
                        await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
                        
                    else:
                        # No lock? Stop motors.
                        await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
                    
                    last_ble_time = now


                # --- 3. TOP-DOWN MAP VISUALIZATION ---
                map_size = 100
                map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
                map_cx = map_size // 2
                scale = 0.5  # 1 pixel on map = 2 cm in reality
                
                # Draw Robot (Blue Dot)
                cv2.circle(map_img, (map_cx, map_size-5), 4, (255, 0, 0), -1)
                
                # Draw Ball (Yellow Dot)
                if curr_z_cm > 0 and is_safe:
                    display_z = predictor.smooth_z if predictor.smooth_z else curr_z_cm
                    mx = int(map_cx + (curr_x_cm * scale))
                    mz = int(map_size - (display_z * scale))
                    if 0 <= mx < map_size and 0 <= mz < map_size:
                        cv2.circle(map_img, (mx, mz), 3, (0, 255, 255), -1)

                # Draw Prediction (Red X)
                if landing:
                    px, pz = landing
                    mx = int(map_cx + (px * scale))
                    mz = int(map_size - (pz * scale))
                    if 0 <= mx < map_size and 0 <= mz < map_size:
                        cv2.line(map_img, (mx-3, mz-3), (mx+3, mz+3), (0, 0, 255), 1)
                        cv2.line(map_img, (mx+3, mz-3), (mx-3, mz+3), (0, 0, 255), 1)

                # Overlay Map on Main Frame
                frame[10:10+map_size, FRAME_WIDTH-10-map_size:FRAME_WIDTH-10] = map_img
                cv2.rectangle(frame, (FRAME_WIDTH-10-map_size, 10), (FRAME_WIDTH-10, 10+map_size), (255, 255, 255), 1)
                
                # Status Text
                cv2.putText(frame, predictor.status, (5, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

                cv2.imshow("Final Catcher", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # --- SAFETY SHUTDOWN ---
            print("Stopping...")
            await client.write_gatt_char(CHARACTERISTIC_UUID, b"M,0,0\n")
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(run_tracker())
    except KeyboardInterrupt:
        print("Interrupted by User.")