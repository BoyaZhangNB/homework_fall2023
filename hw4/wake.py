from wakepy import keep
import time

def wake_for_hours(n_hours):
    """
    Keep the system awake for N hours using wakepy.
    
    Args:
        n_hours: Number of hours to keep the system awake
    """
    seconds = n_hours * 3600
    
    with keep.running():
        print(f"System will stay awake for {n_hours} hours...")
        time.sleep(seconds)
        print("Wake period completed!")

if __name__ == "__main__":
    # Example: keep awake for 2 hours
    wake_for_hours(2)