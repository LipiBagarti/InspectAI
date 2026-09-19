import time

class GPIOMock:
    def __init__(self):
        print("[HARDWARE] Initializing Raspberry Pi GPIO Mock...")
        self.servo_pin = 18
        self.buzzer_pin = 23
        self.led_ok = 17
        self.led_defect = 27
        
        print(f"[HARDWARE] Pins - Servo: {self.servo_pin}, Buzzer: {self.buzzer_pin}, LED_OK: {self.led_ok}, LED_DEFECT: {self.led_defect}")

    def trigger_reject(self):
        print("[HARDWARE] [DEFECT] Activating Reject Servo (Pin 18)...")
        time.sleep(0.1)
        print("[HARDWARE] [BUZZER] BEEP! (Pin 23)")
        print("[HARDWARE] [LED] RED ON (Pin 27)")
        time.sleep(0.2)
        print("[HARDWARE] Resetting servo. LED OFF.")
        
    def trigger_ok(self):
        print("[HARDWARE] [OK] GREEN LED ON (Pin 17)")
        time.sleep(0.1)
        print("[HARDWARE] [OK] GREEN LED OFF")

    def trigger_check(self):
        print("[HARDWARE] [CHECK] YELLOW LED ON (simulated), awaiting human...")
        time.sleep(0.2)
        print("[HARDWARE] [CHECK] YELLOW LED OFF")

if __name__ == "__main__":
    hw = GPIOMock()
    hw.trigger_ok()
    hw.trigger_reject()
    hw.trigger_check()
