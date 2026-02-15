import cv2
import mediapipe as mp
import webbrowser
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=2, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

last_trigger = 0 

print("Controller Online")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    fist_count = 0

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm = hand_lms.landmark
            
            t_tip, i_tip, m_tip = lm[4], lm[8], lm[12]
            
            # Distance Thumb to Index
            pinch_dist = math.hypot(t_tip.x - i_tip.x, t_tip.y - i_tip.y)
            # Distance Thumb to Middle (Should be much larger in a clean pinch)
            m_dist = math.hypot(t_tip.x - m_tip.x, t_tip.y - m_tip.y)

            # CLEAN PINCH CHECK:
            # 1. Thumb and Index are very close (< 0.03)
            # 2. Middle finger is NOT part of the pinch (m_dist > 0.06)
            # 3. Middle and Ring fingers are definitely extended (y-coordinate check)
            middle_is_open = lm[12].y < lm[10].y
            ring_is_open = lm[16].y < lm[14].y

            if pinch_dist < 0.03 and m_dist > 0.07 and middle_is_open and ring_is_open:
                if time.time() - last_trigger > 4:
                    print("CLEAN Pinch Detected! Opening Perplexity...")
                    webbrowser.open("https://www.perplexity.ai/")
                    last_trigger = time.time()

            wrist, m_base = lm[0], lm[9]
            # Use distance from wrist to middle finger knuckle as a scale
            hand_scale = math.hypot(wrist.x - m_base.x, wrist.y - m_base.y)
            # Distance from wrist to middle tip
            extension = math.hypot(wrist.x - m_tip.x, wrist.y - m_tip.y)

            # If the tip of the finger is closer to the wrist than the knuckle is, it's a tight fist
            if extension < hand_scale * 1.1:
                fist_count += 1
                cv2.circle(frame, (int(wrist.x * w), int(wrist.y * h)), 40, (0, 0, 255), 2)

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # --- DOUBLE FIST TRIGGER ---
    if fist_count == 2:
        if time.time() - last_trigger > 5:
            print("DOUBLE FIST DETECTED!")
            webbrowser.open("https://defsnip.vercel.app/")
            last_trigger = time.time()

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()