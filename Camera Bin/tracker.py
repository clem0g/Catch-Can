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

# --- LOGIC SETTINGS ---
# Using the center point you mentioned (177)
TARGET_CENTER_X = 177 
DEADBAND = 20          # How much "wiggle room" for center
STOPPING_RADIUS = 100  # How big object is when it "lands"

async def run_tracker():
    print(f"Scanning for HM-10 ({DEVICE_UUID})...")
    
    # 1. Find the device using the scanner (The method that works)
    device = await BleakScanner.find_device_by_address(DEVICE_UUID, timeout=10.0)
    
    if not device:
        print(f"Device {DEVICE_UUID} not found.")
        return

    print(f"Found {device.name}! Connecting...")

    # 2. Open Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 3. Connect and Start Loop
    async with BleakClient(device) as client:
        print("Connected! Starting Tracking Loop...")
        last_command = None

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Resize to keep logic consistent (Optional, but good for speed)
            frame = cv2.resize(frame, (640, 480))

            # --- VISION LOGIC ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            command = 'S' # Default to Stop if nothing seen

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                
                # Visuals
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

                # --- MOVEMENT DECISIONS ---
                # Check if "Landed" first
                if radius > STOPPING_RADIUS:
                    command = 'S'
                    cv2.putText(frame, "LANDED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Check Steering
                elif x < (TARGET_CENTER_X - DEADBAND):
                    cv2.putText(frame, "LEFT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    command = 'L' # Turn Left
                elif x > (TARGET_CENTER_X + DEADBAND):
                    cv2.putText(frame, "RIGHT", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    command = 'R' # Turn Right
                else:
                    cv2.putText(frame, "FORWARD", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    command = 'F' # Drive Forward (Centered)

            # --- SEND COMMAND ---
            if command != last_command:
                print(f"Sending: {command}")
                try:
                    await client.write_gatt_char(CHARACTERISTIC_UUID, command.encode())
                    last_command = command
                except Exception as e:
                    print(f"BT Error: {e}")

            cv2.imshow("Tracker", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Stop car on exit
        await client.write_gatt_char(CHARACTERISTIC_UUID, b'S')
        cap.release()
        cv2.destroyAllWindows()
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(run_tracker())