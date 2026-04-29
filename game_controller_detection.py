import cv2
import time
import mediapipe as mp

# variable to hold the detection results
latest_result = None

# callback function for asynchronous result
def update_result(result: mp.tasks.vision.ObjectDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

def main():
    # model path 
    model_path = 'PS5controller.tflite' 

    # model setup
    options = mp.tasks.vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        max_results=5,           
        score_threshold=0.03,    # the model only working when threshold is set to 0.03 or below 
        category_allowlist=['ps5-controller'], 
        result_callback=update_result
    )

    # start webcam for detection
    with mp.tasks.vision.ObjectDetector.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting PS5 Controller stream... Press 'q' to quit.")
        
        start_time = time.time()
        controller_first_detected_time = None 
        
        while True:
            success, frame = cap.read()
            if not success:
                print("Error: Failed to capture image.")
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            detector.detect_async(mp_image, timestamp_ms)
            
            controller_detected_this_frame = False 
            
            # draw bounding boxes
            if latest_result:
                for detection in latest_result.detections:
                    bbox = detection.bounding_box
                    start_point = (bbox.origin_x, bbox.origin_y)
                    end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                    
                    cv2.rectangle(frame, start_point, end_point, (255, 0, 255), 2)
                    
                    category = detection.categories[0]
                    category_name = category.category_name
                    probability = round(category.score * 100, 1)
                    
                    if category_name == 'ps5-controller':
                        controller_detected_this_frame = True
                    
                    label = f'{category_name} ({probability}%)'
                    cv2.putText(frame, label, (start_point[0], start_point[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            #raise a message when the controller is detected for 5 continuous seconds
            if controller_detected_this_frame:
                if controller_first_detected_time is None:
                    controller_first_detected_time = time.time()
                else:
                    time_on_screen = time.time() - controller_first_detected_time
                    if time_on_screen >= 5.0:
                        cv2.putText(frame, "Put down the controller!", (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                controller_first_detected_time = None

            cv2.imshow('PS5 Controller Live Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    #clean up resources after everything is done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()