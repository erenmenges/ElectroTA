import serial
import time
import os
import datetime


SERIAL_PORT = 'COM4'
BAUD_RATE = 921600

SAVE_FOLDER = r"C:\Users\jotho\ElectroTA"

os.makedirs(SAVE_FOLDER, exist_ok=True)

print(f"Listening on {SERIAL_PORT}...")
print(f"All photos will be saved to: {SAVE_FOLDER}")
print("Press the button on your ESP32 to take a picture!")

try:
    # open the connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    
    # wait for the ESP32 to finish its automatic reboot
    print("Waiting for ESP32 to boot...")
    time.sleep(2.5) 
    
    # get rid any bytes from old runs
    ser.reset_input_buffer() 
    print("Buffer cleared. Ready for photos!")
    
    image_data = bytearray()
    text_buffer = bytearray()
    latest_voltage = "0.00 V" 
    receiving_image = False
    start_time = 0 

    while True:
        if ser.in_waiting > 0:
            byte = ser.read()
            
            if not receiving_image:
                text_buffer += byte
                
                # did an image just start? (FF D8)
                if len(text_buffer) >= 2 and text_buffer[-2:] == b'\xff\xd8':
                    print("\n Camera clicked! Receiving image data")
                    start_time = time.time()
                    receiving_image = True
                    image_data = bytearray(b'\xff\xd8') 
                    text_buffer = bytearray()
                
                # did a text message just finish?
                elif len(text_buffer) >= 1 and text_buffer[-1:] == b'\n':
                    try:
                        line = text_buffer.decode('utf-8').strip()
                        
                        # if it's probe message, extract the number
                        if "Probe Voltage:" in line:
                            latest_voltage = line.split(":")[-1].strip()
                            print(f" Live Voltage: {latest_voltage}", end="\r") # "\r" makes it overwrite the same line cleanly
                            
                    except UnicodeDecodeError:
                        pass 
                    
                    text_buffer = bytearray() 
                    
            else:
                #we are receiving image instead
                image_data += byte
                
                if len(image_data) >= 2 and image_data[-2:] == b'\xff\xd9':
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"picture_{timestamp}.jpg"
                    filepath = os.path.join(SAVE_FOLDER, filename)
                    
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    
                    end_time = time.time() 
                    time_taken = end_time - start_time
                    
                    print(f"\n Image complete! Saved as: {filename}")
                    print(f"Transfer time: {time_taken:.2f} seconds")
                    
                    # print the voltage that was locked in right before the picture
                    print(f"Voltage at time of photo: {latest_voltage}\n")
                    
                    image_data = bytearray()
                    receiving_image = False

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()