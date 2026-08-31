import cv2
import numpy as np

# --- CONFIGURATION ---
CAMERA_INDEX = 1
LOWER_COLOR = np.array([40, 50, 50])   
UPPER_COLOR = np.array([80, 255, 255])

def run_calibration():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    print("--- CALIBRATION MODE ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

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
                # Draw Visuals
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # --- DISPLAY THE KEY DATA ---
                # We make this text BIG so you can't miss it
                cv2.putText(frame, f"PIXEL RADIUS: {radius:.2f}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Calibration: Read the Radius", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration()