import numpy as np
from scipy.signal import find_peaks

class AdaptiveSegmentation:
    def __init__(self, window_size=50, threshold=1.5):
        self.window_size = window_size
        self.threshold = threshold

    def detect_change_points(self, data):
        """Detect change points using a simple sliding window approach."""
        n = len(data)
        change_points = []
        
        for i in range(self.window_size, n - self.window_size):
            left_window = data[i - self.window_size:i]
            right_window = data[i:i + self.window_size]
            
            left_mean = np.mean(left_window)
            right_mean = np.mean(right_window)
            
            if abs(left_mean - right_mean) > self.threshold * np.std(data):
                change_points.append(i)
        
        return change_points

    def segment_time_series(self, data):
        """Segment the time series based on detected change points."""
        change_points = self.detect_change_points(data)
        segments = []
        start = 0
        
        for cp in change_points:
            segments.append(data[start:cp])
            start = cp
        
        segments.append(data[start:])
        return segments

    def adaptive_window_size(self, data):
        """Adapt window size based on signal characteristics."""
        # Use peak detection to estimate signal frequency
        peaks, _ = find_peaks(data)
        if len(peaks) > 1:
            avg_peak_distance = np.mean(np.diff(peaks))
            self.window_size = max(int(avg_peak_distance), 10)
        
    def run(self, data):
        """Run the adaptive segmentation algorithm."""
        self.adaptive_window_size(data)
        return self.segment_time_series(data)

# Example usage
if __name__ == "__main__":
    # Generate a sample time series
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(2 * np.pi * t) + np.random.normal(0, 0.1, 1000)
    signal[500:] += 2  # Add a step change

    # Run adaptive segmentation
    segmenter = AdaptiveSegmentation()
    segments = segmenter.run(signal)

    print(f"Number of segments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"Segment {i+1} length: {len(seg)}")

    # Plotting (optional, requires matplotlib)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal, label='Original Signal')
    for i, seg in enumerate(segments):
        plt.plot(t[sum(len(s) for s in segments[:i]):sum(len(s) for s in segments[:i+1])], 
                 seg, label=f'Segment {i+1}')
    plt.legend()
    plt.title('Adaptive Time Series Segmentation')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()