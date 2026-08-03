# Audio Visualizer

This is an audio visualization application built using Rust, EGUI, and the `cpal` crate for audio input. The application displays an animated spectrum of audio frequencies in real-time.

### Features

- Real-time audio frequency visualization
- Playback controls: Play/Pause, Previous, Next tracks
- Album art integration via MPD (Media Player Daemon) metadata
- Smoothed bars and peak hold indicators for better visibility

### Dependencies

- Rust compiler (nightly or stable)
- EGUI library
- `cpal` crate for audio input
- `realfft` crate for FFT calculations
- `urlencoding` crate for URL decoding (optional)

### Installation

1. Ensure you have Rust installed on your system.
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-visualizer.git
   ```
3. Navigate to the project directory:
   ```bash
   cd audio-visualizer
   ```
4. Build the application:
   ```bash
   cargo build
   ```

### Running

1. Ensure MPD is running and accessible on your system.
2. Run the application:
   ```bash
   cargo run
   ```

### Configuration

- The application uses MPD metadata to fetch track details, so ensure MPD is configured correctly and accessible from the machine running the application.

### Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
5. Open a pull request on the original repository.

## License
This project is licensed under the MIT License.
---