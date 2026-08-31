import asyncio
import numpy as np
import cv2
from bleak import BleakClient

# --- BLUETOOTH CONFIGURATION ---
# Your specific UUID
DEVICE_UUID = "033F2D12-9FF3-419C-95CA-5F746FA68B8E"
# HM-10 Characteristic
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

# --- CAMERA & VISION CONFIGURATION ---
CAMERA_INDEX = 1  # OBS Virtual Camera
LOWER_COLOR = np.array([35, 50, 50]) 
UPPER_COLOR = np.array([85, 255, 255])

async def run():
    print(f"Attempting to connect to HM-10 ({DEVICE_UUID})...")
    
    # Initialize Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"Camera found: {frame_width}x{frame_height}")

    try:
        # Start Bluetooth Connection
        async with BleakClient(DEVICE_UUID) as client:
            print(f"Connected to {DEVICE_UUID}!")
            print("Starting Tracking Loop... Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # --- IMAGE PROCESSING ---
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

                            # Calculate offsets (Your specific logic)
                            offset_x = int(cp_x - (frame_width/2))
                            # Using int() to ensure clean numbers for Arduino
                            offset_y = int(-(cp_y - (frame_height * 0.8)))

                            # Prepare string for Arduino (e.g., "50, 100\n")
                            cur_pos = f"{offset_x}, {offset_y}\n"
                            
                            # --- SEND VIA BLUETOOTH ---
                            # We use 'await' here so it doesn't block the video
                            try:
                                await client.write_gatt_char(CHARACTERISTIC_UUID, cur_pos.encode('utf-8'))
                                # Optional: Print to console to verify it's sending
                                # print(f"Sent: {cur_pos.strip()}")
                                print(f"{cp_x},{cp_y}")
                            except Exception as e:
                                print(f"Bluetooth Send Error: {e}")

                            # Draw Visuals
                            cv2.circle(frame, (cp_x, cp_y), 10, (0, 255, 0), -1)
                            cv2.putText(frame, f"Sending: {cp_x}, {cp_y}", 
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Show the video feed
                cv2.imshow("Trash Can Vision", frame)

                # Check for 'q' key to quit
                # waitKey(1) is small enough to not block the async loop noticeably
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quitting...")
                    break
                    
    except Exception as e:
        print(f"Connection or Runtime Error: {e}")
        
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed.")

if __name__ == "__main__":
    asyncio.run(run())