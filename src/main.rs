use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use egui::ecolor::Hsva;
use egui::{
    Align2, Color32, CornerRadius, FontId, Pos2, Rect, Shape, Stroke, TextureHandle,
    TextureOptions, epaint::Mesh, pos2, vec2,
};
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

// ============================================================================
// CONSTANTS & DSP CONFIGURATION
// ============================================================================
pub const NUM_MEL_BARS: usize = 64;
pub const FFT_SIZE: usize = 1024;
pub const WAVEFORM_SIZE: usize = 512;
pub const SAMPLE_RATE: f32 = 44100.0;
pub const MIN_FREQ: f32 = 25.0;
pub const MAX_FREQ: f32 = 14000.0;
pub const MIN_DB: f32 = -60.0;
pub const MAX_DB: f32 = 0.0;
const ALBUM_ART_SIZE: u32 = 160;

const MIN_BAR_WIDTH: f32 = 4.0;
const BAR_GAP: f32 = 3.0;
const MIN_SPECTRUM_HEIGHT: f32 = 80.0;

const ATTACK_RATE: f32 = 100.0;
const RELEASE_RATE: f32 = 35.0;
const PEAK_RELEASE_RATE: f32 = 4.5;

// ============================================================================
// STEP 1 & 2: TELEMETRY PAYLOAD (IPC BROADCAST DATA)
// ============================================================================
#[derive(Clone, Debug)]
pub struct DspFrame {
    pub mel_bars: [f32; NUM_MEL_BARS],
    pub waveform: [f32; WAVEFORM_SIZE],
    pub is_beat: bool,
    pub beat_intensity: f32,
    pub spectral_flux: f32,
    pub agc_gain: f32,
    pub bass_energy: f32,
    pub treble_energy: f32,
}

impl Default for DspFrame {
    fn default() -> Self {
        Self {
            mel_bars: [0.0; NUM_MEL_BARS],
            waveform: [0.0; WAVEFORM_SIZE],
            is_beat: false,
            beat_intensity: 0.0,
            spectral_flux: 0.0,
            agc_gain: 1.0,
            bass_energy: 0.0,
            treble_energy: 0.0,
        }
    }
}

pub struct TrackUpdate {
    pub title: String,
    pub artist: String,
    pub album: String,
    pub art: Option<egui::ColorImage>,
}

pub struct PlaybackUpdate {
    pub is_playing: bool,
    pub position: Duration,
    pub length: Option<Duration>,
    pub volume: Option<f64>,
}

pub enum MprisMessage {
    Track(TrackUpdate),
    Playback(PlaybackUpdate),
}

// ============================================================================
// STEP 1: PSYCHOACOUSTIC MEL-SCALE FILTER BANK (WITH PEAK POOLING)
// ============================================================================
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

pub struct MelFilterBank {
    pub num_bars: usize,
    pub fft_size: usize,
    filter_weights: Vec<Vec<(usize, f32)>>,
}

impl MelFilterBank {
    pub fn new(
        num_bars: usize,
        fft_size: usize,
        sample_rate: f32,
        min_freq: f32,
        max_freq: f32,
    ) -> Self {
        let min_mel = hz_to_mel(min_freq);
        let max_mel = hz_to_mel(max_freq);

        let mut mel_points = Vec::with_capacity(num_bars + 2);
        for i in 0..=(num_bars + 1) {
            let mel = min_mel + (max_mel - min_mel) * (i as f32) / ((num_bars + 1) as f32);
            mel_points.push(mel_to_hz(mel));
        }

        let num_bins = fft_size / 2 + 1;
        let bin_width = sample_rate / (fft_size as f32);

        let mut filter_weights = Vec::with_capacity(num_bars);

        for m in 1..=num_bars {
            let left_hz = mel_points[m - 1];
            let center_hz = mel_points[m];
            let right_hz = mel_points[m + 1];

            let mut weights = Vec::new();
            for bin in 0..num_bins {
                let freq = bin as f32 * bin_width;

                if freq >= left_hz && freq <= center_hz {
                    let weight = (freq - left_hz) / (center_hz - left_hz);
                    if weight > 0.001 {
                        weights.push((bin, weight));
                    }
                } else if freq > center_hz && freq <= right_hz {
                    let weight = (right_hz - freq) / (right_hz - center_hz);
                    if weight > 0.001 {
                        weights.push((bin, weight));
                    }
                }
            }

            filter_weights.push(weights);
        }

        Self {
            num_bars,
            fft_size,
            filter_weights,
        }
    }

    pub fn compute(&self, spectrum: &[Complex<f32>], out_bars: &mut [f32]) {
        let norm_factor = 1.0 / (self.fft_size as f32 * 0.5);
        for (m, weights) in self.filter_weights.iter().enumerate() {
            let mut peak_mag = 0.0f32;
            for &(bin, weight) in weights {
                if bin < spectrum.len() {
                    let mag = spectrum[bin].norm() * norm_factor * (0.6 + 0.4 * weight);
                    peak_mag = peak_mag.max(mag);
                }
            }

            // High dynamic range (-60 dB floor) with logarithmic loudness
            let freq_tilt = 1.0 + (m as f32 / (self.num_bars as f32 - 1.0).max(1.0)) * 1.5;
            let energy_boosted = (peak_mag * freq_tilt).max(1e-9);

            let db = 20.0 * energy_boosted.log10();
            let norm = ((db - MIN_DB) / (MAX_DB - MIN_DB)).clamp(0.0, 1.0);
            out_bars[m] = norm;
        }
    }
}

// ============================================================================
// STEP 1: ONSET / BEAT DETECTION & SPECTRAL FLUX
// ============================================================================
pub struct BeatDetector {
    prev_magnitudes: Vec<f32>,
    flux_history: [f32; 43],
    history_idx: usize,
    decay_energy: f32,
}

impl BeatDetector {
    pub fn new(num_bins: usize) -> Self {
        Self {
            prev_magnitudes: vec![0.0; num_bins],
            flux_history: [0.0; 43],
            history_idx: 0,
            decay_energy: 0.0,
        }
    }

    pub fn detect(&mut self, spectrum: &[Complex<f32>]) -> (bool, f32, f32) {
        let num_bins = spectrum.len().min(self.prev_magnitudes.len());
        let norm_factor = 1.0 / (self.prev_magnitudes.len() as f32 * 0.5);
        let mut flux = 0.0f32;
        let bass_bins = (num_bins as f32 * 0.12) as usize;

        for i in 0..num_bins {
            let mag = spectrum[i].norm() * norm_factor;
            let diff = mag - self.prev_magnitudes[i];
            if diff > 0.0 {
                let weight = if i < bass_bins { 3.0 } else { 1.0 };
                flux += diff * weight;
            }
            self.prev_magnitudes[i] = mag;
        }

        self.flux_history[self.history_idx] = flux;
        self.history_idx = (self.history_idx + 1) % self.flux_history.len();

        let avg_flux: f32 =
            self.flux_history.iter().sum::<f32>() / (self.flux_history.len() as f32);
        let variance: f32 = self
            .flux_history
            .iter()
            .map(|f| (f - avg_flux).powi(2))
            .sum::<f32>()
            / (self.flux_history.len() as f32);
        let threshold = avg_flux + variance.sqrt() * 1.5;

        let is_beat = flux > threshold && flux > self.decay_energy && flux > 0.02;
        let intensity = if is_beat {
            ((flux - threshold) / (threshold.max(0.01))).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.decay_energy = (self.decay_energy * 0.85).max(flux * 0.9);

        (is_beat, intensity, flux)
    }
}

// ============================================================================
// STEP 2: MULTI-THREADED REAL-TIME AUDIO CAPTURE & DSP THREAD
// ============================================================================
fn spawn_audio_pipeline(tx_dsp: mpsc::Sender<DspFrame>) -> cpal::Stream {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .or_else(|| host.default_input_device())
        .expect("No audio input or output device available");

    let supported_config = device
        .default_output_config()
        .or_else(|_| device.default_input_config())
        .expect("Failed to get audio stream config");

    let channels = supported_config.channels() as usize;
    let config = supported_config.config();

    let (tx_audio, rx_audio) = mpsc::channel::<Vec<f32>>();
    let tx_audio_in = tx_audio.clone();
    let tx_audio_out = tx_audio;

    thread::spawn(move || {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(FFT_SIZE);
        let mut indata = r2c.make_input_vec();
        let mut spectrum = r2c.make_output_vec();

        let mut acc: Vec<f32> = Vec::with_capacity(FFT_SIZE * 4);
        let mel_bank = MelFilterBank::new(NUM_MEL_BARS, FFT_SIZE, SAMPLE_RATE, MIN_FREQ, MAX_FREQ);
        let mut beat_detector = BeatDetector::new(FFT_SIZE / 2 + 1);

        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * n as f32 / (FFT_SIZE as f32 - 1.0)).cos())
            })
            .collect();

        let hop_size = FFT_SIZE / 2;

        for samples in rx_audio {
            acc.extend_from_slice(&samples);

            while acc.len() >= FFT_SIZE {
                let work_chunk = &acc[..FFT_SIZE];

                let mut waveform = [0.0f32; WAVEFORM_SIZE];
                let wave_step = FFT_SIZE / WAVEFORM_SIZE;
                for i in 0..WAVEFORM_SIZE {
                    let s = work_chunk[i * wave_step] * 3.6;
                    waveform[i] = if s.abs() > 0.95 {
                        s.signum() * (0.95 + (s.abs() - 0.95).tanh() * 0.05)
                    } else {
                        s
                    };
                }

                for (i, (&s, &w)) in work_chunk.iter().zip(window.iter()).enumerate() {
                    indata[i] = s * w;
                }

                r2c.process(&mut indata, &mut spectrum).ok();

                let mut mel_bars = [0.0f32; NUM_MEL_BARS];
                mel_bank.compute(&spectrum, &mut mel_bars);

                let (is_beat, beat_intensity, spectral_flux) = beat_detector.detect(&spectrum);

                let num_bins = spectrum.len();
                let norm_factor = 1.0 / (FFT_SIZE as f32 * 0.5);
                let bass_bins = (num_bins as f32 * 0.1) as usize;
                let treble_bins_start = (num_bins as f32 * 0.6) as usize;

                let bass_energy: f32 = spectrum[..bass_bins]
                    .iter()
                    .map(|c| c.norm() * norm_factor)
                    .sum::<f32>()
                    / bass_bins.max(1) as f32;
                let treble_energy: f32 = spectrum[treble_bins_start..]
                    .iter()
                    .map(|c| c.norm() * norm_factor)
                    .sum::<f32>()
                    / (num_bins - treble_bins_start).max(1) as f32;

                let frame = DspFrame {
                    mel_bars,
                    waveform,
                    is_beat,
                    beat_intensity,
                    spectral_flux,
                    agc_gain: 1.0,
                    bass_energy: (bass_energy * 6.0).clamp(0.0, 1.0),
                    treble_energy: (treble_energy * 10.0).clamp(0.0, 1.0),
                };

                if tx_dsp.send(frame).is_err() {
                    return;
                }

                acc.drain(..hop_size);
            }
        }
    });

    let err_fn = |err| eprintln!("Audio stream error: {}", err);
    let stream = device
        .build_input_stream(
            config.clone(),
            move |data: &[f32], _: &_| {
                let mono: Vec<f32> = if channels <= 1 {
                    data.to_vec()
                } else {
                    data.chunks_exact(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect()
                };
                tx_audio_in.send(mono).ok();
            },
            err_fn,
            None,
        )
        .or_else(|_| {
            device.build_output_stream(
                config,
                move |data: &mut [f32], _: &_| {
                    let mono: Vec<f32> = if channels <= 1 {
                        data.to_vec()
                    } else {
                        data.chunks_exact(channels)
                            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                            .collect()
                    };
                    tx_audio_out.send(mono).ok();
                },
                err_fn,
                None,
            )
        })
        .expect("Failed to build input or loopback audio stream");

    stream.play().expect("Failed to start audio stream");
    stream
}

// ============================================================================
// STEP 3 & 4: NATIVE WGPU GPU PIPELINE & WGSL SHADERS
// ============================================================================
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DspUniformsGpu {
    pub time: f32,
    pub beat_intensity: f32,
    pub bass_energy: f32,
    pub treble_energy: f32,
    pub spectral_flux: f32,
    pub style_index: f32,
    pub resolution: [f32; 2],
    pub bg_top: [f32; 4],
    pub bg_bottom: [f32; 4],
    pub led_low: [f32; 4],
    pub led_mid: [f32; 4],
    pub led_high: [f32; 4],
    pub text_accent: [f32; 4],
    pub mel_bars: [f32; 64],
}

pub const VISUALIZER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct DspUniforms {
    time: f32,
    beat_intensity: f32,
    bass_energy: f32,
    treble_energy: f32,
    spectral_flux: f32,
    style_index: f32,
    resolution: vec2<f32>,
    bg_top: vec4<f32>,
    bg_bottom: vec4<f32>,
    led_low: vec4<f32>,
    led_mid: vec4<f32>,
    led_high: vec4<f32>,
    text_accent: vec4<f32>,
    mel_bars_0: vec4<f32>,
    mel_bars_1: vec4<f32>,
    mel_bars_2: vec4<f32>,
    mel_bars_3: vec4<f32>,
    mel_bars_4: vec4<f32>,
    mel_bars_5: vec4<f32>,
    mel_bars_6: vec4<f32>,
    mel_bars_7: vec4<f32>,
    mel_bars_8: vec4<f32>,
    mel_bars_9: vec4<f32>,
    mel_bars_10: vec4<f32>,
    mel_bars_11: vec4<f32>,
    mel_bars_12: vec4<f32>,
    mel_bars_13: vec4<f32>,
    mel_bars_14: vec4<f32>,
    mel_bars_15: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: DspUniforms;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    let pos = positions[in_vertex_index];
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

fn get_bar(idx: u32) -> f32 {
    let vec_idx = idx / 4u;
    let comp_idx = idx % 4u;
    var v: vec4<f32>;
    switch (vec_idx) {
        case 0u: { v = uniforms.mel_bars_0; }
        case 1u: { v = uniforms.mel_bars_1; }
        case 2u: { v = uniforms.mel_bars_2; }
        case 3u: { v = uniforms.mel_bars_3; }
        case 4u: { v = uniforms.mel_bars_4; }
        case 5u: { v = uniforms.mel_bars_5; }
        case 6u: { v = uniforms.mel_bars_6; }
        case 7u: { v = uniforms.mel_bars_7; }
        case 8u: { v = uniforms.mel_bars_8; }
        case 9u: { v = uniforms.mel_bars_9; }
        case 10u: { v = uniforms.mel_bars_10; }
        case 11u: { v = uniforms.mel_bars_11; }
        case 12u: { v = uniforms.mel_bars_12; }
        case 13u: { v = uniforms.mel_bars_13; }
        case 14u: { v = uniforms.mel_bars_14; }
        default: { v = uniforms.mel_bars_15; }
    }
    if (comp_idx == 0u) { return v.x; }
    if (comp_idx == 1u) { return v.y; }
    if (comp_idx == 2u) { return v.z; }
    return v.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;

    // 1. Dynamic Atmosphere & Palette Gradient
    var color = mix(uniforms.bg_top.rgb, uniforms.bg_bottom.rgb, uv.y);

    // 2. Audio-Reactive Background Spectrum Plumes
    let bar_idx = u32(clamp(uv.x * 64.0, 0.0, 63.0));
    let bar_val = get_bar(bar_idx);
    let plume_y = exp(-abs(uv.y - (1.0 - bar_val * 0.85)) * 14.0) * (bar_val * 0.35);
    let plume_col = mix(uniforms.led_low.rgb, uniforms.led_mid.rgb, uv.x);
    color += plume_col * plume_y;

    // 3. Subtle Audio Particle Stardust
    let p_grid = sin(uv * 70.0 + vec2<f32>(uniforms.time * 2.0, -uniforms.time * 1.5));
    let p_dot = smoothstep(0.96, 1.0, p_grid.x * p_grid.y);
    color += uniforms.led_high.rgb * (p_dot * (0.12 + uniforms.treble_energy * 0.6));

    // 4. Scanlines (subtle modern retro texture without screen warping)
    let scanline = sin(uv.y * uniforms.resolution.y * 1.2) * 0.03;
    color -= vec3<f32>(scanline);

    // 5. Luminescence Falloff / Vignette
    let vignette = smoothstep(0.85, 0.25, length(uv - 0.5));
    color *= vignette * 0.35 + 0.65;

    return vec4<f32>(color, 1.0);
}
"#;

pub struct GpuVisualizerPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GpuVisualizerPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Visualizer Shader"),
            source: wgpu::ShaderSource::Wgsl(VISUALIZER_WGSL.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visualizer Uniforms"),
            size: std::mem::size_of::<DspUniformsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Visualizer Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Visualizer Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Visualizer Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            ..Default::default()
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Visualizer Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
        }
    }
}

pub struct GpuCallback {
    pub uniforms: DspUniformsGpu,
}

impl CallbackTrait for GpuCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = callback_resources.get::<GpuVisualizerPipeline>() {
            queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        if let Some(res) = callback_resources.get::<GpuVisualizerPipeline>() {
            render_pass.set_pipeline(&res.pipeline);
            render_pass.set_bind_group(0, &res.bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }
    }
}

// ============================================================================
// APP PALETTES & STYLES
// ============================================================================
#[derive(Clone, Copy)]
pub struct AppPalette {
    bg_top: Color32,
    bg_bottom: Color32,
    panel_bg: Color32,
    panel_border: Color32,
    button_rest: Color32,
    button_hover: Color32,
    button_click: Color32,
    text_muted: Color32,
    text_primary: Color32,
    text_accent: Color32,
    led_low: Color32,
    led_mid: Color32,
    led_high: Color32,
    record_label: Color32,
}

impl Default for AppPalette {
    fn default() -> Self {
        Self {
            bg_top: Color32::from_rgb(42, 45, 52),
            bg_bottom: Color32::from_rgb(27, 28, 32),
            panel_bg: Color32::from_rgb(34, 37, 42),
            panel_border: Color32::from_rgb(110, 115, 125),
            button_rest: Color32::from_rgb(45, 50, 60),
            button_hover: Color32::from_rgb(55, 65, 80),
            button_click: Color32::from_rgb(35, 40, 50),
            text_muted: Color32::from_rgb(150, 160, 175),
            text_primary: Color32::from_rgb(230, 240, 248),
            text_accent: Color32::from_rgb(75, 163, 195),
            led_low: Color32::from_rgb(23, 86, 118),
            led_mid: Color32::from_rgb(75, 163, 195),
            led_high: Color32::from_rgb(204, 230, 244),
            record_label: Color32::from_rgb(29, 47, 111),
        }
    }
}

fn color_to_vec4(c: Color32) -> [f32; 4] {
    [
        c.r() as f32 / 255.0,
        c.g() as f32 / 255.0,
        c.b() as f32 / 255.0,
        c.a() as f32 / 255.0,
    ]
}

fn hue_sat_from_image(img: &egui::ColorImage) -> Option<(f32, f32)> {
    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;
    let mut count = 0.0;
    let mut r_fall = 0.0;
    let mut g_fall = 0.0;
    let mut b_fall = 0.0;
    let mut fall_count = 0.0;

    for p in &img.pixels {
        let r = p.r() as f32;
        let g = p.g() as f32;
        let b = p.b() as f32;
        let hsva = Hsva::from(*p);

        if hsva.v > 0.2 && hsva.v < 0.8 && hsva.s > 0.3 {
            r_sum += r;
            g_sum += g;
            b_sum += b;
            count += 1.0;
        }
        r_fall += r;
        g_fall += g;
        b_fall += b;
        fall_count += 1.0;
    }

    let avg_color = if count > 0.0 {
        Color32::from_rgb(
            (r_sum / count) as u8,
            (g_sum / count) as u8,
            (b_sum / count) as u8,
        )
    } else if fall_count > 0.0 {
        Color32::from_rgb(
            (r_fall / fall_count) as u8,
            (g_fall / fall_count) as u8,
            (b_fall / fall_count) as u8,
        )
    } else {
        return None;
    };

    let base_hsva = Hsva::from(avg_color);
    Some((base_hsva.h, base_hsva.s.max(0.15)))
}

impl AppPalette {
    fn curve(h: f32, s: f32, v_top: f32, v_bot: f32) -> Self {
        let s = s.clamp(0.15, 0.9);
        Self {
            bg_top: Hsva::new(h, s * 0.35, v_top, 1.0).into(),
            bg_bottom: Hsva::new(h, s * 0.40, v_bot, 1.0).into(),
            panel_bg: Hsva::new(h, s * 0.25, (v_top + v_bot) * 0.5, 1.0).into(),
            panel_border: Hsva::new(h, s * 0.30, 0.55, 1.0).into(),
            button_rest: Hsva::new(h, s * 0.30, 0.22, 1.0).into(),
            button_hover: Hsva::new(h, s * 0.40, 0.32, 1.0).into(),
            button_click: Hsva::new(h, s * 0.30, 0.16, 1.0).into(),
            text_muted: Hsva::new(h, s * 0.20, 0.65, 1.0).into(),
            text_primary: Hsva::new(h, s * 0.10, 0.96, 1.0).into(),
            text_accent: Hsva::new(h, (s * 1.4).min(1.0), 0.85, 1.0).into(),
            led_low: Hsva::new(h, s, 0.45, 1.0).into(),
            led_mid: Hsva::new((h + 0.05) % 1.0, (s * 1.3).min(1.0), 0.80, 1.0).into(),
            led_high: Hsva::new((h + 0.1) % 1.0, s * 0.5, 0.98, 1.0).into(),
            record_label: Hsva::new(h, s, 0.35, 1.0).into(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VisualizerStyle {
    LedEqualizer,
    Oscilloscope,
    RadialPulse,
    Bars3D,
    MirrorSpectrum,
    NeonMountain,
}

impl VisualizerStyle {
    const ALL: [VisualizerStyle; 6] = [
        VisualizerStyle::LedEqualizer,
        VisualizerStyle::Oscilloscope,
        VisualizerStyle::RadialPulse,
        VisualizerStyle::Bars3D,
        VisualizerStyle::MirrorSpectrum,
        VisualizerStyle::NeonMountain,
    ];

    fn index(self) -> f32 {
        match self {
            Self::LedEqualizer => 0.0,
            Self::Oscilloscope => 1.0,
            Self::RadialPulse => 2.0,
            Self::Bars3D => 3.0,
            Self::MirrorSpectrum => 4.0,
            Self::NeonMountain => 5.0,
        }
    }

    fn next(self) -> Self {
        let idx = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    fn prev(self) -> Self {
        let idx = Self::ALL.iter().position(|s| *s == self).unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    fn label(self) -> &'static str {
        match self {
            Self::LedEqualizer => "LED EQUALIZER",
            Self::Oscilloscope => "OSCILLOSCOPE",
            Self::RadialPulse => "RADIAL PULSE",
            Self::Bars3D => "3D SPECTROGRAM",
            Self::MirrorSpectrum => "MIRROR SPECTRUM",
            Self::NeonMountain => "NEON MOUNTAIN",
        }
    }

    fn build_palette(self, tint: Option<(f32, f32)>) -> AppPalette {
        if let Some((h, s)) = tint {
            let (bg1, bg2) = match self {
                Self::LedEqualizer => (0.22, 0.12),
                Self::Oscilloscope => (0.15, 0.05),
                Self::RadialPulse => (0.25, 0.08),
                Self::Bars3D => (0.30, 0.16),
                Self::MirrorSpectrum => (0.20, 0.06),
                Self::NeonMountain => (0.25, 0.10),
            };
            return AppPalette::curve(h, s, bg1, bg2);
        }

        match self {
            Self::LedEqualizer => AppPalette::curve(0.55, 0.5, 0.22, 0.12),
            Self::Oscilloscope => AppPalette::curve(0.33, 0.85, 0.15, 0.05),
            Self::RadialPulse => AppPalette::curve(0.83, 0.85, 0.25, 0.08),
            Self::Bars3D => AppPalette::curve(0.58, 0.35, 0.30, 0.16),
            Self::MirrorSpectrum => AppPalette::curve(0.11, 0.75, 0.20, 0.06),
            Self::NeonMountain => AppPalette::curve(0.9, 0.6, 0.25, 0.10),
        }
    }
}

fn format_time(d: Duration) -> String {
    let secs = d.as_secs();
    format!("{}:{:02}", secs / 60, secs % 60)
}

fn catmull_rom(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let f0 = -0.5 * t3 + t2 - 0.5 * t;
    let f1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let f3 = 0.5 * t3 - 0.5 * t2;
    pos2(
        p0.x * f0 + p1.x * f1 + p2.x * f2 + p3.x * f3,
        p0.y * f0 + p1.y * f1 + p2.y * f2 + p3.y * f3,
    )
}

fn exponential_smooth(current: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let alpha = 1.0 - (-rate * dt).exp();
    current + (target - current) * alpha
}

fn downsample_bars(bars: &[f32], display_count: usize) -> Vec<f32> {
    let n = bars.len();
    if display_count >= n {
        return bars.to_vec();
    }
    let display_count = display_count.max(1);
    let mut out = vec![0.0f32; display_count];
    for (i, &value) in bars.iter().enumerate() {
        let bin = (i * display_count) / n;
        out[bin] = out[bin].max(value);
    }
    out
}

// ============================================================================
// MPRIS & MEDIA INTEGRATION (LINUX & WINDOWS)
// ============================================================================
#[cfg(target_os = "linux")]
fn load_album_art(uri: &str) -> Option<egui::ColorImage> {
    let img = if let Some(mut path_str) = uri.strip_prefix("file://") {
        if path_str.starts_with("//") {
            path_str = &path_str[1..];
        }
        if path_str.starts_with("localhost/") {
            path_str = &path_str[10..];
        }
        let clean_path = urlencoding::decode(path_str)
            .unwrap_or_else(|_| path_str.into())
            .into_owned();
        image::open(clean_path).ok()?
    } else if uri.starts_with("http://") || uri.starts_with("https://") {
        let bytes = reqwest::blocking::get(uri).ok()?.bytes().ok()?;
        image::load_from_memory(&bytes).ok()?
    } else if uri.starts_with("data:image/") {
        let parts: Vec<&str> = uri.split(',').collect();
        if parts.len() == 2 {
            use base64::{Engine as _, engine::general_purpose};
            let bytes = general_purpose::STANDARD.decode(parts[1]).ok()?;
            image::load_from_memory(&bytes).ok()?
        } else {
            return None;
        }
    } else {
        return None;
    };

    let img = img.resize_to_fill(
        ALBUM_ART_SIZE,
        ALBUM_ART_SIZE,
        image::imageops::FilterType::Lanczos3,
    );
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    Some(egui::ColorImage::from_rgba_unmultiplied(
        [w, h],
        &rgba.into_raw(),
    ))
}

#[cfg(target_os = "linux")]
fn find_best_player(finder: &mpris::PlayerFinder) -> Option<mpris::Player> {
    let mut players = finder.find_all().ok()?;
    if players.is_empty() {
        return None;
    }

    let playing_idx = players
        .iter()
        .position(|p| matches!(p.get_playback_status(), Ok(mpris::PlaybackStatus::Playing)));
    let paused_idx = players
        .iter()
        .position(|p| matches!(p.get_playback_status(), Ok(mpris::PlaybackStatus::Paused)));

    playing_idx
        .or(paused_idx)
        .map(|idx| players.swap_remove(idx))
}

#[cfg(target_os = "linux")]
fn spawn_mpris_thread(tx_mpris: mpsc::Sender<MprisMessage>) {
    thread::spawn(move || {
        let mut last_id: Option<String> = None;
        if let Ok(finder) = mpris::PlayerFinder::new() {
            loop {
                if let Some(player) = find_best_player(&finder) {
                    let status = player
                        .get_playback_status()
                        .unwrap_or(mpris::PlaybackStatus::Stopped);
                    let is_playing = status == mpris::PlaybackStatus::Playing;
                    let position = player.get_position().unwrap_or_default();
                    let volume = player.get_volume().ok();

                    let metadata = player.get_metadata().ok();
                    let length = metadata.as_ref().and_then(|m| m.length());

                    tx_mpris
                        .send(MprisMessage::Playback(PlaybackUpdate {
                            is_playing,
                            position,
                            length,
                            volume,
                        }))
                        .ok();

                    if let Some(metadata) = metadata {
                        let id =
                            metadata
                                .track_id()
                                .map(|id| id.to_string())
                                .unwrap_or_else(|| {
                                    format!("{:?}{:?}", metadata.title(), metadata.artists())
                                });

                        if last_id.as_deref() != Some(id.as_str()) {
                            last_id = Some(id);
                            let title = metadata.title().unwrap_or("Unknown title").to_string();
                            let artist = metadata
                                .artists()
                                .map(|a| a.join(", "))
                                .unwrap_or_else(|| "Unknown artist".to_string());
                            let album = metadata.album_name().unwrap_or("").to_string();
                            let art = metadata.art_url().and_then(load_album_art);

                            tx_mpris
                                .send(MprisMessage::Track(TrackUpdate {
                                    title,
                                    artist,
                                    album,
                                    art,
                                }))
                                .ok();
                        }
                    }
                } else if last_id.is_some() {
                    last_id = None;
                    tx_mpris
                        .send(MprisMessage::Track(TrackUpdate {
                            title: String::new(),
                            artist: String::new(),
                            album: String::new(),
                            art: None,
                        }))
                        .ok();
                    tx_mpris
                        .send(MprisMessage::Playback(PlaybackUpdate {
                            is_playing: false,
                            position: Duration::default(),
                            length: None,
                            volume: None,
                        }))
                        .ok();
                }
                thread::sleep(Duration::from_millis(200));
            }
        }
    });
}

#[cfg(target_os = "windows")]
fn spawn_mpris_thread(tx_mpris: mpsc::Sender<MprisMessage>) {
    use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
    use windows::Storage::Streams::DataReader;

    thread::spawn(move || {
        let manager = match GlobalSystemMediaTransportControlsSessionManager::RequestAsync() {
            Ok(op) => match op.get() {
                Ok(m) => m,
                Err(_) => return,
            },
            Err(_) => return,
        };

        let mut last_id = String::new();

        loop {
            if let Ok(session) = manager.GetCurrentSession() {
                if let Ok(playback) = session.GetPlaybackInfo() {
                    let status = playback.PlaybackStatus().unwrap_or_default().0;
                    let is_playing = status == 4;

                    let mut position = Duration::default();
                    let mut length = None;

                    if let Ok(timeline) = session.GetTimelineProperties() {
                        let position_ticks =
                            timeline.Position().unwrap_or_default().Duration as u64;
                        let last_updated =
                            timeline.LastUpdatedTime().unwrap_or_default().UniversalTime as u64;

                        let unix_now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_micros() as u64
                            * 10;
                        let win_now = unix_now + 116_444_736_000_000_000;

                        let time_elapsed = if is_playing && win_now > last_updated {
                            win_now - last_updated
                        } else {
                            0
                        };

                        position = Duration::from_micros((position_ticks + time_elapsed) / 10);
                        length = Some(Duration::from_micros(
                            (timeline.EndTime().unwrap_or_default().Duration / 10) as u64,
                        ));
                    }

                    tx_mpris
                        .send(MprisMessage::Playback(PlaybackUpdate {
                            is_playing,
                            position,
                            length,
                            volume: None,
                        }))
                        .ok();
                }

                if let Ok(media_props) =
                    session.TryGetMediaPropertiesAsync().and_then(|op| op.get())
                {
                    let title = media_props.Title().unwrap_or_default().to_string();
                    let artist = media_props.Artist().unwrap_or_default().to_string();
                    let album = media_props.AlbumTitle().unwrap_or_default().to_string();

                    let track_id = format!("{}-{}-{}", title, artist, album);
                    if track_id != last_id {
                        last_id = track_id;
                        let mut art = None;

                        if let Ok(thumb_ref) = media_props.Thumbnail() {
                            if let Ok(stream) = thumb_ref.OpenReadAsync().and_then(|op| op.get()) {
                                let size = stream.Size().unwrap_or(0) as u32;
                                if size > 0 {
                                    if let Ok(reader) = DataReader::CreateDataReader(&stream) {
                                        if reader.LoadAsync(size).and_then(|op| op.get()).is_ok() {
                                            let mut bytes = vec![0u8; size as usize];
                                            if reader.ReadBytes(&mut bytes).is_ok() {
                                                if let Ok(img) = image::load_from_memory(&bytes) {
                                                    let img = img.resize_to_fill(
                                                        160,
                                                        160,
                                                        image::imageops::FilterType::Lanczos3,
                                                    );
                                                    let rgba = img.to_rgba8();
                                                    art = Some(
                                                        egui::ColorImage::from_rgba_unmultiplied(
                                                            [
                                                                rgba.width() as usize,
                                                                rgba.height() as usize,
                                                            ],
                                                            &rgba.into_raw(),
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        tx_mpris
                            .send(MprisMessage::Track(TrackUpdate {
                                title,
                                artist,
                                album,
                                art,
                            }))
                            .ok();
                    }
                }
            } else {
                tx_mpris
                    .send(MprisMessage::Playback(PlaybackUpdate {
                        is_playing: false,
                        position: Duration::default(),
                        length: None,
                        volume: None,
                    }))
                    .ok();
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

#[cfg(not(any(target_os = "windows", target_os = "linux")))]
fn spawn_mpris_thread(_tx_mpris: mpsc::Sender<MprisMessage>) {
    // Stub for other platforms
}

// ============================================================================
// VISUALIZER APP IMPLEMENTATION
// ============================================================================
pub struct VisualizerApp {
    rx_dsp: mpsc::Receiver<DspFrame>,
    rx_mpris: mpsc::Receiver<MprisMessage>,
    latest_dsp: DspFrame,
    smoothed_bars: [f32; NUM_MEL_BARS],
    peak_bars: [f32; NUM_MEL_BARS],
    screen_flash: f32,
    camera_shake: f32,
    album_art: Option<TextureHandle>,
    title: String,
    artist: String,
    album: String,
    record_angle: f32,
    pulse_rotation: f32,
    text_scroll: f32,
    is_playing: bool,
    position: Duration,
    length: Option<Duration>,
    volume: Option<f64>,
    current_palette: AppPalette,
    style: VisualizerStyle,
    album_tint: Option<(f32, f32)>,
    has_wgpu: bool,
    start_time: Instant,
}

fn with_alpha(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

fn paint_dynamic_gradient_background(painter: &egui::Painter, rect: Rect, palette: &AppPalette) {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.left_top(), palette.bg_top);
    mesh.colored_vertex(rect.right_top(), palette.bg_top);
    mesh.colored_vertex(rect.left_bottom(), palette.bg_bottom);
    mesh.colored_vertex(rect.right_bottom(), palette.bg_bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    painter.add(Shape::mesh(mesh));
}

fn draw_hardware_button(ui: &mut egui::Ui, text: &str, palette: &AppPalette) -> egui::Response {
    let size = vec2(46.0, 28.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter();

    let is_clicked = response.is_pointer_button_down_on();
    let is_hovered = response.hovered();

    let bg_color = if is_clicked {
        palette.button_click
    } else if is_hovered {
        palette.button_hover
    } else {
        palette.button_rest
    };

    painter.rect_filled(rect, CornerRadius::same(3), bg_color);
    painter.rect_stroke(
        rect,
        CornerRadius::same(3),
        Stroke::new(1.0, palette.panel_border),
        egui::StrokeKind::Inside,
    );

    let text_color = if is_hovered {
        palette.text_primary
    } else {
        palette.text_muted
    };
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        FontId::monospace(12.0),
        text_color,
    );

    response
}

fn draw_marquee_text(
    ui: &mut egui::Ui,
    text: &str,
    font: FontId,
    color: Color32,
    offset: f32,
    max_width: f32,
) {
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, color);
    let text_width = galley.size().x;
    let (rect, _) = ui.allocate_exact_size(
        vec2(max_width, galley.size().y.max(16.0)),
        egui::Sense::hover(),
    );
    let painter = ui.painter();

    if text_width <= max_width {
        painter.galley(rect.left_top(), galley, color);
    } else {
        let scroll_max = text_width + 40.0;
        let current_offset = offset % scroll_max;

        painter.with_clip_rect(rect).galley(
            rect.left_top() - vec2(current_offset, 0.0),
            galley.clone(),
            color,
        );

        if current_offset > (text_width + 40.0 - max_width) {
            painter.with_clip_rect(rect).galley(
                rect.left_top() + vec2(scroll_max - current_offset, 0.0),
                galley,
                color,
            );
        }
    }
}

impl VisualizerApp {
    fn draw_now_playing(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let sleeve_size = 140.0;
            let record_radius = sleeve_size / 2.0;
            let protruding_amount = record_radius * 0.85;

            let (rect, _) = ui.allocate_exact_size(
                vec2(sleeve_size + protruding_amount, sleeve_size),
                egui::Sense::hover(),
            );
            let painter = ui.painter();

            let sleeve_rect = Rect::from_min_size(rect.left_top(), vec2(sleeve_size, sleeve_size));
            let record_center = pos2(
                sleeve_rect.right() - record_radius + protruding_amount,
                sleeve_rect.center().y,
            );

            painter.circle_filled(record_center, record_radius, Color32::from_rgb(18, 18, 18));
            painter.circle_stroke(
                record_center,
                record_radius * 0.85,
                Stroke::new(1.0, Color32::from_rgb(35, 35, 35)),
            );
            painter.circle_stroke(
                record_center,
                record_radius * 0.70,
                Stroke::new(1.0, Color32::from_rgb(25, 25, 25)),
            );

            let label_radius = record_radius * 0.42;
            painter.circle_filled(
                record_center,
                label_radius,
                self.current_palette.record_label,
            );

            let rot_dir = vec2(self.record_angle.cos(), self.record_angle.sin());
            let marker_color = self.current_palette.text_primary;

            painter.circle_stroke(
                record_center,
                label_radius * 0.8,
                Stroke::new(1.0, marker_color),
            );
            let dot_pos = record_center + rot_dir * (label_radius * 0.5);
            painter.circle_filled(dot_pos, label_radius * 0.15, marker_color);

            painter.rect_filled(
                sleeve_rect,
                CornerRadius::same(4),
                self.current_palette.bg_bottom,
            );

            if let Some(tex) = &self.album_art {
                painter.image(
                    tex.id(),
                    sleeve_rect,
                    Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                    Color32::WHITE,
                );
            } else {
                painter.text(
                    sleeve_rect.center(),
                    Align2::CENTER_CENTER,
                    "\u{266A}",
                    FontId::proportional(60.0),
                    self.current_palette.text_muted,
                );
            }

            painter.rect_stroke(
                sleeve_rect,
                CornerRadius::same(4),
                Stroke::new(1.0, self.current_palette.panel_border),
                egui::StrokeKind::Inside,
            );

            ui.add_space(24.0);

            ui.vertical(|ui| {
                ui.add_space(10.0);
                ui.label(
                    egui::RichText::new("NOW PLAYING")
                        .size(11.0)
                        .monospace()
                        .color(self.current_palette.text_muted),
                );

                let title = if self.title.is_empty() {
                    "NOTHING PLAYING"
                } else {
                    &self.title
                };
                let available_text_width = ui.available_width() - 20.0;

                draw_marquee_text(
                    ui,
                    &title.to_uppercase(),
                    FontId::monospace(20.0),
                    self.current_palette.text_primary,
                    self.text_scroll,
                    available_text_width,
                );

                if !self.artist.is_empty() {
                    draw_marquee_text(
                        ui,
                        &self.artist.to_uppercase(),
                        FontId::monospace(13.0),
                        self.current_palette.text_accent,
                        self.text_scroll * 0.7,
                        available_text_width,
                    );
                }

                ui.add_space(6.0);

                let progress_rect_height = 16.0;
                let (prog_rect, response) = ui.allocate_exact_size(
                    vec2(available_text_width, progress_rect_height),
                    egui::Sense::click_and_drag(),
                );
                let prog_painter = ui.painter();

                let time_str = format_time(self.position);
                let rem_str = if let Some(len) = self.length {
                    let rem = len.saturating_sub(self.position);
                    format!("-{}", format_time(rem))
                } else {
                    "--:--".to_string()
                };

                prog_painter.text(
                    prog_rect.left_center(),
                    Align2::LEFT_CENTER,
                    time_str,
                    FontId::monospace(11.0),
                    self.current_palette.text_muted,
                );
                prog_painter.text(
                    prog_rect.right_center(),
                    Align2::RIGHT_CENTER,
                    rem_str,
                    FontId::monospace(11.0),
                    self.current_palette.text_muted,
                );

                let bar_left = prog_rect.left() + 38.0;
                let bar_right = prog_rect.right() - 42.0;
                let bar_y = prog_rect.center().y;

                prog_painter.line_segment(
                    [pos2(bar_left, bar_y), pos2(bar_right, bar_y)],
                    Stroke::new(2.0, self.current_palette.button_rest),
                );

                if let Some(len) = self.length {
                    if len.as_secs_f32() > 0.0 {
                        let ratio =
                            (self.position.as_secs_f32() / len.as_secs_f32()).clamp(0.0, 1.0);
                        let playhead_x = bar_left + (bar_right - bar_left) * ratio;

                        prog_painter.line_segment(
                            [pos2(bar_left, bar_y), pos2(playhead_x, bar_y)],
                            Stroke::new(2.0, self.current_palette.text_accent),
                        );
                        prog_painter.circle_filled(
                            pos2(playhead_x, bar_y),
                            3.5,
                            self.current_palette.text_primary,
                        );

                        if response.clicked() || response.dragged() {
                            if let Some(pos) = response.interact_pointer_pos() {
                                let scrub_ratio = ((pos.x - bar_left) / (bar_right - bar_left)).clamp(0.0, 1.0);
                                let new_pos = Duration::from_secs_f32(len.as_secs_f32() * scrub_ratio);
                                
                                #[cfg(target_os = "linux")]
                                std::thread::spawn(move || {
                                    if let Ok(finder) = mpris::PlayerFinder::new() {
                                        if let Some(player) = find_best_player(&finder) {
                                            if let Ok(metadata) = player.get_metadata() {
                                                if let Some(track_id) = metadata.track_id() {
                                                    let _ = player.set_position(track_id.clone(), &new_pos);
                                                }
                                            }
                                        }
                                    }
                                });

                                #[cfg(target_os = "windows")]
                                std::thread::spawn(move || {
                                    use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                                    if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                        if let Ok(session) = manager.GetCurrentSession() {
                                            let ticks = (new_pos.as_micros() * 10) as i64;
                                            session.TryChangePlaybackPositionAsync(ticks).ok();
                                        }
                                    }
                                });
                            }
                        }
                    }
                }

                ui.add_space(8.0);

                // Physical Playback Transport Buttons
                ui.horizontal(|ui| {
                    if draw_hardware_button(ui, "\u{23EE}", &self.current_palette).clicked() {
                        #[cfg(target_os = "linux")]
                        std::thread::spawn(|| {
                            if let Ok(finder) = mpris::PlayerFinder::new() {
                                if let Some(player) = find_best_player(&finder) {
                                    player.previous().ok();
                                }
                            }
                        });
                        #[cfg(target_os = "windows")]
                        std::thread::spawn(|| {
                            use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                            if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                if let Ok(session) = manager.GetCurrentSession() {
                                    session.TrySkipPreviousAsync().ok();
                                }
                            }
                        });
                    }

                    let play_icon = if self.is_playing { "\u{23F8}" } else { "\u{25B6}" };
                    if draw_hardware_button(ui, play_icon, &self.current_palette).clicked() {
                        #[cfg(target_os = "linux")]
                        std::thread::spawn(|| {
                            if let Ok(finder) = mpris::PlayerFinder::new() {
                                if let Some(player) = find_best_player(&finder) {
                                    player.play_pause().ok();
                                }
                            }
                        });
                        #[cfg(target_os = "windows")]
                        std::thread::spawn(|| {
                            use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                            if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                if let Ok(session) = manager.GetCurrentSession() {
                                    session.TryTogglePlayPauseAsync().ok();
                                }
                            }
                        });
                    }

                    if draw_hardware_button(ui, "\u{23ED}", &self.current_palette).clicked() {
                        #[cfg(target_os = "linux")]
                        std::thread::spawn(|| {
                            if let Ok(finder) = mpris::PlayerFinder::new() {
                                if let Some(player) = find_best_player(&finder) {
                                    player.next().ok();
                                }
                            }
                        });
                        #[cfg(target_os = "windows")]
                        std::thread::spawn(|| {
                            use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                            if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                if let Ok(session) = manager.GetCurrentSession() {
                                    session.TrySkipNextAsync().ok();
                                }
                            }
                        });
                    }

                    if let Some(vol) = self.volume {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add_space(4.0);
                            let (rect, response) = ui.allocate_exact_size(vec2(60.0, 16.0), egui::Sense::click_and_drag());
                            let painter = ui.painter();
                            
                            let bg_rect = Rect::from_center_size(rect.center(), vec2(rect.width(), 4.0));
                            painter.rect_filled(bg_rect, CornerRadius::same(2), self.current_palette.button_rest);
                            
                            let vol_width = rect.width() * (vol as f32).clamp(0.0, 1.0);
                            let fg_rect = Rect::from_min_size(bg_rect.left_top(), vec2(vol_width, 4.0));
                            painter.rect_filled(fg_rect, CornerRadius::same(2), self.current_palette.text_accent);
                            
                            let handle_pos = pos2(bg_rect.left() + vol_width, bg_rect.center().y);
                            painter.circle_filled(handle_pos, 4.0, self.current_palette.text_primary);
                            
                            ui.add_space(4.0);
                            ui.label(egui::RichText::new("\u{1F50A}").color(self.current_palette.text_muted).size(12.0));
                            
                            if response.clicked() || response.dragged() {
                                if let Some(pos) = response.interact_pointer_pos() {
                                    let new_vol = ((pos.x - bg_rect.left()) / bg_rect.width()).clamp(0.0, 1.0) as f64;
                                    
                                    #[cfg(target_os = "linux")]
                                    std::thread::spawn(move || {
                                        if let Ok(finder) = mpris::PlayerFinder::new() {
                                            if let Some(player) = find_best_player(&finder) {
                                                player.set_volume(new_vol).ok();
                                            }
                                        }
                                    });
                                }
                            }
                        });
                    }
                });
            });
        });
    }

    fn draw_spectrum(&self, ui: &mut egui::Ui, height: f32) {
        let desired_size = vec2(ui.available_width(), height.max(MIN_SPECTRUM_HEIGHT));
        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());

        // --- STEP 3 & 4: NATIVE GPU AMBIENT SHADER PASS ---
        if self.has_wgpu {
            let mut mel_bars = [0.0f32; 64];
            for (dst, &src) in mel_bars.iter_mut().zip(self.smoothed_bars.iter()) {
                *dst = src.clamp(0.0, 1.0);
            }

            let uniforms = DspUniformsGpu {
                time: self.start_time.elapsed().as_secs_f32(),
                beat_intensity: 0.0, // No flash inside visualization
                bass_energy: self.latest_dsp.bass_energy,
                treble_energy: self.latest_dsp.treble_energy,
                spectral_flux: self.latest_dsp.spectral_flux,
                style_index: self.style.index(),
                resolution: [rect.width(), rect.height()],
                bg_top: color_to_vec4(self.current_palette.bg_top),
                bg_bottom: color_to_vec4(self.current_palette.bg_bottom),
                led_low: color_to_vec4(self.current_palette.led_low),
                led_mid: color_to_vec4(self.current_palette.led_mid),
                led_high: color_to_vec4(self.current_palette.led_high),
                text_accent: color_to_vec4(self.current_palette.text_accent),
                mel_bars,
            };

            let cb = egui_wgpu::Callback::new_paint_callback(rect, GpuCallback { uniforms });
            ui.painter().add(cb);
        }

        // --- CRISP FOREGROUND VISUALIZER LAYER ---
        let painter = ui.painter();

        match self.style {
            VisualizerStyle::LedEqualizer => {
                let available_width = rect.width().max(1.0);
                let max_bars_that_fit = (((available_width + BAR_GAP) / (MIN_BAR_WIDTH + BAR_GAP))
                    .floor() as usize)
                    .max(1);
                let n = max_bars_that_fit.min(self.smoothed_bars.len());

                let values = downsample_bars(&self.smoothed_bars, n);
                let peaks = downsample_bars(&self.peak_bars, n);

                let bar_width =
                    ((available_width - BAR_GAP * (n as f32 - 1.0)) / n as f32).max(1.0);
                let segment_gap = 2.0;
                let desired_segment_height = (bar_width * 0.5).clamp(4.0, 12.0);
                let segments = (((rect.height() + segment_gap)
                    / (desired_segment_height + segment_gap))
                    .floor() as usize)
                    .clamp(6, 64);
                let segment_height =
                    (rect.height() - (segments - 1) as f32 * segment_gap) / segments as f32;

                for (i, (&value, &peak)) in values.iter().zip(peaks.iter()).enumerate() {
                    let x = rect.left() + i as f32 * (bar_width + BAR_GAP);
                    let active_segments =
                        (value.clamp(0.0, 1.0) * segments as f32).round() as usize;
                    let peak_segment = (peak.clamp(0.0, 1.0) * segments as f32).round() as usize;

                    for s in 0..segments {
                        let y = rect.bottom()
                            - (s as f32 + 1.0) * segment_height
                            - s as f32 * segment_gap;
                        let seg_rect =
                            Rect::from_min_max(pos2(x, y), pos2(x + bar_width, y + segment_height));

                        let is_active = s < active_segments;
                        let is_peak = s == peak_segment;

                        let base_color = if s < (segments / 2) {
                            self.current_palette.led_low
                        } else if s < (segments * 4 / 5) {
                            self.current_palette.led_mid
                        } else {
                            self.current_palette.led_high
                        };

                        let color = if is_active || is_peak {
                            base_color
                        } else {
                            base_color.linear_multiply(0.12)
                        };
                        painter.rect_filled(seg_rect, CornerRadius::same(1), color);

                        if is_active {
                            painter.rect_filled(
                                seg_rect.expand(2.5),
                                CornerRadius::same(2),
                                with_alpha(base_color, 25),
                            );
                        }
                    }
                }
            }

            VisualizerStyle::Oscilloscope => {
                let mid_y = rect.center().y;
                let amp = rect.height() * 0.28;

                let grid_color = with_alpha(self.current_palette.led_low, 35);
                for i in 1..8 {
                    let gy = rect.top() + rect.height() * (i as f32 / 8.0);
                    painter.line_segment(
                        [pos2(rect.left(), gy), pos2(rect.right(), gy)],
                        Stroke::new(1.0, grid_color),
                    );
                }
                for i in 1..12 {
                    let gx = rect.left() + rect.width() * (i as f32 / 12.0);
                    painter.line_segment(
                        [pos2(gx, rect.top()), pos2(gx, rect.bottom())],
                        Stroke::new(1.0, grid_color),
                    );
                }

                let count = self.latest_dsp.waveform.len().max(2);
                let step_x = rect.width() / (count - 1) as f32;

                let points: Vec<Pos2> = self
                    .latest_dsp
                    .waveform
                    .iter()
                    .enumerate()
                    .map(|(i, &sample)| {
                        let x = rect.left() + i as f32 * step_x;
                        let y = mid_y - sample * amp;
                        pos2(x, y.clamp(rect.top(), rect.bottom()))
                    })
                    .collect();

                let mut smooth_points = Vec::with_capacity(points.len() * 4);
                for i in 0..points.len().saturating_sub(1) {
                    let p0 = if i > 0 { points[i - 1] } else { points[i] };
                    let p1 = points[i];
                    let p2 = points[i + 1];
                    let p3 = if i + 2 < points.len() { points[i + 2] } else { points[i + 1] };

                    for step in 0..4 {
                        let t = step as f32 / 4.0;
                        smooth_points.push(catmull_rom(p0, p1, p2, p3, t));
                    }
                }
                if let Some(&last) = points.last() {
                    smooth_points.push(last);
                }

                if smooth_points.len() > 2 {
                    let mut fill_mesh = Mesh::default();
                    for &p in &smooth_points {
                        fill_mesh.colored_vertex(p, with_alpha(self.current_palette.led_high, 0));
                        fill_mesh.colored_vertex(pos2(p.x, rect.bottom()), with_alpha(self.current_palette.led_low, 30));
                    }
                    for i in 0..smooth_points.len() - 1 {
                        let idx = (i * 2) as u32;
                        fill_mesh.add_triangle(idx, idx + 1, idx + 2);
                        fill_mesh.add_triangle(idx + 1, idx + 3, idx + 2);
                    }
                    painter.add(Shape::mesh(fill_mesh));
                }

                for thickness in [10.0, 4.0, 1.5] {
                    let alpha = if thickness > 5.0 { 30 } else if thickness > 2.0 { 100 } else { 255 };
                    let color = if thickness > 2.0 { self.current_palette.led_mid } else { self.current_palette.led_high };
                    painter.add(Shape::line(
                        smooth_points.clone(),
                        Stroke::new(thickness, with_alpha(color, alpha)),
                    ));
                }
            }

            VisualizerStyle::RadialPulse => {
                let center = rect.center();
                let max_radius = rect.width().min(rect.height()) * 0.52;
                let inner_radius = max_radius * 0.32;

                let num_rays = NUM_MEL_BARS;
                let angle_step = std::f32::consts::TAU / num_rays as f32;

                let core_r = inner_radius * (0.80 + self.latest_dsp.bass_energy * 0.40);
                
                painter.circle_filled(center, core_r, self.current_palette.led_low);
                painter.circle_stroke(
                    center,
                    core_r * 1.15,
                    Stroke::new(3.0, with_alpha(self.current_palette.led_mid, 150)),
                );
                painter.circle_stroke(
                    center,
                    core_r * 1.30,
                    Stroke::new(1.0, with_alpha(self.current_palette.led_high, 80)),
                );

                let n = num_rays / 2;
                let values = downsample_bars(&self.smoothed_bars, n);
                
                let mut mirrored_values = Vec::with_capacity(num_rays);
                for i in 0..n {
                    mirrored_values.push(values[i]);
                }
                for i in (0..n).rev() {
                    mirrored_values.push(values[i]);
                }

                let mut raw_points = Vec::with_capacity(num_rays);
                for (i, &val) in mirrored_values.iter().enumerate() {
                    let angle = self.pulse_rotation + i as f32 * angle_step;
                    let dir = vec2(angle.cos(), angle.sin());
                    let bar_len = (max_radius - inner_radius) * val.clamp(0.0, 1.0);
                    raw_points.push(center + dir * (inner_radius + bar_len));
                }

                let mut smooth_points = Vec::new();
                for i in 0..num_rays {
                    let p0 = raw_points[(i + num_rays - 1) % num_rays];
                    let p1 = raw_points[i];
                    let p2 = raw_points[(i + 1) % num_rays];
                    let p3 = raw_points[(i + 2) % num_rays];

                    for step in 0..4 {
                        let t = step as f32 / 4.0;
                        smooth_points.push(catmull_rom(p0, p1, p2, p3, t));
                    }
                }

                let mut blob_mesh = Mesh::default();
                blob_mesh.colored_vertex(center, with_alpha(self.current_palette.led_mid, 20));
                for &p in &smooth_points {
                    blob_mesh.colored_vertex(p, with_alpha(self.current_palette.led_low, 120));
                }
                for i in 1..=smooth_points.len() {
                    let next_i = if i == smooth_points.len() { 1 } else { i + 1 };
                    blob_mesh.add_triangle(0, i as u32, next_i as u32);
                }
                painter.add(Shape::mesh(blob_mesh));

                let mut closed_line_pts = smooth_points.clone();
                if let Some(&first) = smooth_points.first() {
                    closed_line_pts.push(first);
                }
                
                painter.add(Shape::line(
                    closed_line_pts.clone(),
                    Stroke::new(6.0, with_alpha(self.current_palette.led_mid, 60)),
                ));
                painter.add(Shape::line(
                    closed_line_pts,
                    Stroke::new(2.0, self.current_palette.led_high),
                ));
            }

            VisualizerStyle::MirrorSpectrum => {
                let mid_y = rect.center().y;
                let half_h = rect.height() * 0.48;

                let n = self.smoothed_bars.len().min(64);
                let values = downsample_bars(&self.smoothed_bars, n);
                let bar_w = ((rect.width() - BAR_GAP * (n as f32 - 1.0)) / n as f32).max(1.0);

                for (i, &val) in values.iter().enumerate() {
                    let x = rect.left() + i as f32 * (bar_w + BAR_GAP);
                    let bar_h = (val * half_h).clamp(2.0, half_h);

                    let top_rect =
                        Rect::from_min_max(pos2(x, mid_y - bar_h), pos2(x + bar_w, mid_y - 1.0));
                    let bot_rect =
                        Rect::from_min_max(pos2(x, mid_y + 1.0), pos2(x + bar_w, mid_y + bar_h));

                    let color = if i < n / 3 {
                        self.current_palette.led_low
                    } else if i < n * 2 / 3 {
                        self.current_palette.led_mid
                    } else {
                        self.current_palette.led_high
                    };

                    painter.rect_filled(top_rect, CornerRadius::same(1), color);
                    painter.rect_filled(
                        bot_rect,
                        CornerRadius::same(1),
                        color.linear_multiply(0.75),
                    );
                }

                painter.line_segment(
                    [pos2(rect.left(), mid_y), pos2(rect.right(), mid_y)],
                    Stroke::new(1.0, self.current_palette.text_muted),
                );
            }

            VisualizerStyle::Bars3D => {
                let n = self.smoothed_bars.len().min(48);
                let values = downsample_bars(&self.smoothed_bars, n);
                
                let center_x = rect.center().x;
                let horizon_y = rect.top() + rect.height() * 0.3;
                let fov = 300.0;
                
                let project = |x: f32, y: f32, z: f32| -> Pos2 {
                    let z_scale = fov / (fov + z).max(1.0);
                    pos2(
                        center_x + (x - center_x) * z_scale,
                        horizon_y + (y - horizon_y) * z_scale,
                    )
                };

                let floor_y = rect.bottom() - horizon_y - 20.0;
                
                let grid_z_step = 40.0;
                let grid_color = with_alpha(self.current_palette.led_low, 40);
                for i in 0..15 {
                    let z = i as f32 * grid_z_step;
                    let p_left = project(rect.left() - 200.0, horizon_y + floor_y, z);
                    let p_right = project(rect.right() + 200.0, horizon_y + floor_y, z);
                    painter.line_segment([p_left, p_right], Stroke::new(1.0, grid_color));
                }
                
                let spacing = (rect.width() * 0.8) / n as f32;
                let start_x = rect.center().x - (rect.width() * 0.4);
                let max_h = rect.height() * 0.5;

                for (i, &val) in values.iter().enumerate() {
                    let x = start_x + i as f32 * spacing;
                    let h = (val * max_h).clamp(2.0, max_h);
                    let z = 50.0 + (i as f32 % 2.0) * 10.0;
                    let bar_w = spacing * 0.8;
                    let depth = 15.0;

                    let v0 = (x, horizon_y + floor_y, z);
                    let v1 = (x + bar_w, horizon_y + floor_y, z);
                    let v2 = (x + bar_w, horizon_y + floor_y - h, z);
                    let v3 = (x, horizon_y + floor_y - h, z);
                    
                    let v4 = (x, horizon_y + floor_y, z + depth);
                    let v5 = (x + bar_w, horizon_y + floor_y, z + depth);
                    let v6 = (x + bar_w, horizon_y + floor_y - h, z + depth);
                    let v7 = (x, horizon_y + floor_y - h, z + depth);

                    let p0 = project(v0.0, v0.1, v0.2);
                    let p1 = project(v1.0, v1.1, v1.2);
                    let p2 = project(v2.0, v2.1, v2.2);
                    let p3 = project(v3.0, v3.1, v3.2);
                    let p6 = project(v6.0, v6.1, v6.2);
                    let p7 = project(v7.0, v7.1, v7.2);

                    let h_norm = (h / max_h).clamp(0.0, 1.0);
                    let color_bottom = self.current_palette.led_low;
                    let color_top = if h_norm > 0.6 { self.current_palette.led_high } else { self.current_palette.led_mid };

                    let pr2 = project(v2.0, horizon_y + floor_y + h, v2.2);
                    let pr3 = project(v3.0, horizon_y + floor_y + h, v3.2);
                    let mut refl_mesh = Mesh::default();
                    refl_mesh.colored_vertex(p0, with_alpha(color_bottom, 40));
                    refl_mesh.colored_vertex(p1, with_alpha(color_bottom, 40));
                    refl_mesh.colored_vertex(pr2, with_alpha(color_top, 5));
                    refl_mesh.colored_vertex(pr3, with_alpha(color_top, 5));
                    refl_mesh.add_triangle(0, 1, 2);
                    refl_mesh.add_triangle(0, 2, 3);
                    painter.add(Shape::mesh(refl_mesh));

                    let mut front_mesh = Mesh::default();
                    front_mesh.colored_vertex(p0, color_bottom);
                    front_mesh.colored_vertex(p1, color_bottom);
                    front_mesh.colored_vertex(p2, color_top);
                    front_mesh.colored_vertex(p3, color_top);
                    front_mesh.add_triangle(0, 1, 2);
                    front_mesh.add_triangle(0, 2, 3);
                    painter.add(Shape::mesh(front_mesh));

                    let mut top_mesh = Mesh::default();
                    top_mesh.colored_vertex(p3, color_top);
                    top_mesh.colored_vertex(p2, color_top);
                    top_mesh.colored_vertex(p6, color_top.linear_multiply(0.7));
                    top_mesh.colored_vertex(p7, color_top.linear_multiply(0.7));
                    top_mesh.add_triangle(0, 1, 2);
                    top_mesh.add_triangle(0, 2, 3);
                    painter.add(Shape::mesh(top_mesh));

                    let mut side_mesh = Mesh::default();
                    side_mesh.colored_vertex(p1, color_bottom.linear_multiply(0.5));
                    side_mesh.colored_vertex(project(v5.0, v5.1, v5.2), color_bottom.linear_multiply(0.3));
                    side_mesh.colored_vertex(p6, color_top.linear_multiply(0.5));
                    side_mesh.colored_vertex(p2, color_top.linear_multiply(0.8));
                    side_mesh.add_triangle(0, 1, 2);
                    side_mesh.add_triangle(0, 2, 3);
                    painter.add(Shape::mesh(side_mesh));
                }
            }

            VisualizerStyle::NeonMountain => {
                let mid_y = rect.bottom() - 10.0;
                let amp = rect.height() * 0.7;

                let n = self.smoothed_bars.len().min(48);
                let values = downsample_bars(&self.smoothed_bars, n);
                
                let mut mirrored_values = Vec::with_capacity(n * 2 - 1);
                for i in (1..n).rev() {
                    mirrored_values.push(values[i]);
                }
                for i in 0..n {
                    mirrored_values.push(values[i]);
                }
                
                let step_x = rect.width() / (mirrored_values.len() - 1) as f32;

                let points: Vec<Pos2> = mirrored_values
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| {
                        let x = rect.left() + i as f32 * step_x;
                        let y = mid_y - (val * amp);
                        pos2(x, y.clamp(rect.top(), rect.bottom()))
                    })
                    .collect();

                let mut smooth_points = Vec::with_capacity(points.len() * 4);
                for i in 0..points.len().saturating_sub(1) {
                    let p0 = if i > 0 { points[i - 1] } else { pos2(points[i].x - step_x, mid_y) };
                    let p1 = points[i];
                    let p2 = points[i + 1];
                    let p3 = if i + 2 < points.len() { points[i + 2] } else { pos2(points[i + 1].x + step_x, mid_y) };

                    for step in 0..4 {
                        let t = step as f32 / 4.0;
                        smooth_points.push(catmull_rom(p0, p1, p2, p3, t));
                    }
                }
                if let Some(&last) = points.last() {
                    smooth_points.push(last);
                }

                if smooth_points.len() > 2 {
                    let mut mountain_mesh = Mesh::default();
                    for &p in &smooth_points {
                        let h_norm = ((mid_y - p.y) / amp).clamp(0.0, 1.0);
                        let c_top = self.current_palette.led_mid.linear_multiply(0.8 + 0.2 * h_norm);
                        
                        mountain_mesh.colored_vertex(p, with_alpha(c_top, 220));
                        mountain_mesh.colored_vertex(pos2(p.x, mid_y), with_alpha(self.current_palette.led_low, 20));
                    }
                    for i in 0..smooth_points.len() - 1 {
                        let idx = (i * 2) as u32;
                        mountain_mesh.add_triangle(idx, idx + 1, idx + 2);
                        mountain_mesh.add_triangle(idx + 1, idx + 3, idx + 2);
                    }
                    painter.add(Shape::mesh(mountain_mesh));

                    painter.add(Shape::line(
                        smooth_points,
                        Stroke::new(3.0, self.current_palette.led_high),
                    ));
                }
            }
        }
    }

    fn refresh_palette(&mut self) {
        self.current_palette = self.style.build_palette(self.album_tint);
    }
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dt = ui.input(|i| i.stable_dt).clamp(1.0 / 240.0, 0.1);

        if self.is_playing {
            self.record_angle += dt * 1.8;
            self.text_scroll += dt * 35.0;
            self.position += Duration::from_secs_f32(dt);
        }
        self.pulse_rotation += dt * (0.8 + self.latest_dsp.bass_energy * 3.5);

        let mut latest_frame = None;
        while let Ok(frame) = self.rx_dsp.try_recv() {
            if frame.is_beat {
                self.screen_flash = frame.beat_intensity;
                self.camera_shake = frame.beat_intensity * 6.0;
            }
            latest_frame = Some(frame);
        }

        if let Some(mut frame) = latest_frame {
            for s in frame.waveform.iter_mut() {
                *s = (*s * 1.8).clamp(-2.5, 2.5);
            }
            for b in frame.mel_bars.iter_mut() {
                *b = (*b * 1.5).clamp(0.0, 1.0);
            }

            for (curr, &target) in self.smoothed_bars.iter_mut().zip(frame.mel_bars.iter()) {
                let rate = if target > *curr {
                    ATTACK_RATE
                } else {
                    RELEASE_RATE
                };
                *curr = exponential_smooth(*curr, target, rate, dt);
            }
            self.latest_dsp = frame;
        }

        self.screen_flash = (self.screen_flash - dt * 2.0).max(0.0);
        self.camera_shake = (self.camera_shake - dt * 20.0).max(0.0);

        for (peak, &current) in self.peak_bars.iter_mut().zip(self.smoothed_bars.iter()) {
            let decayed = *peak * (-PEAK_RELEASE_RATE * dt).exp();
            *peak = current.max(decayed);
        }

        while let Ok(msg) = self.rx_mpris.try_recv() {
            match msg {
                MprisMessage::Track(update) => {
                    self.title = update.title;
                    self.artist = update.artist;
                    self.album = update.album;
                    self.album_tint = update.art.as_ref().and_then(hue_sat_from_image);
                    self.refresh_palette();
                    self.album_art = update.art.map(|color_image| {
                        ui.ctx()
                            .load_texture("album_art", color_image, TextureOptions::LINEAR)
                    });
                }
                MprisMessage::Playback(p) => {
                    self.is_playing = p.is_playing;
                    self.position = p.position;
                    self.length = p.length;
                    if let Some(vol) = p.volume {
                        self.volume = Some(vol);
                    }
                }
            }
        }

        let bg_flash_color = if self.screen_flash > 0.05 {
            with_alpha(
                self.current_palette.led_high,
                (self.screen_flash * 65.0) as u8,
            )
        } else {
            Color32::TRANSPARENT
        };

        // Outer window background: flashes on beat drops
        paint_dynamic_gradient_background(ui.painter(), ui.max_rect(), &self.current_palette);
        if self.screen_flash > 0.05 {
            ui.painter()
                .rect_filled(ui.max_rect(), CornerRadius::ZERO, bg_flash_color);
        }

        let outer_margin = egui::Margin::same(24);
        let background = egui::Frame::default().inner_margin(outer_margin);

        background.show(ui, |ui| {
            let panel_rect = ui.available_rect_before_wrap();
            ui.painter().rect_filled(
                panel_rect,
                CornerRadius::same(6),
                self.current_palette.panel_bg,
            );
            ui.painter().rect_stroke(
                panel_rect,
                CornerRadius::same(6),
                Stroke::new(1.0, self.current_palette.panel_border),
                egui::StrokeKind::Inside,
            );

            egui::Frame::default()
                .inner_margin(egui::Margin::same(20))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let header_title = if self.has_wgpu {
                            "STEREO VISUALIZER"
                        } else {
                            "STEREO VISUALIZER"
                        };
                        ui.label(
                            egui::RichText::new(header_title)
                                .color(self.current_palette.text_primary)
                                .monospace()
                                .size(16.0)
                                .strong(),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(vec2(12.0, 12.0), egui::Sense::hover());
                            ui.painter().circle_filled(
                                rect.center(),
                                4.0,
                                self.current_palette.text_accent,
                            );
                            ui.label(
                                egui::RichText::new("POWER")
                                    .monospace()
                                    .size(10.0)
                                    .color(self.current_palette.text_muted),
                            );
                        });
                    });

                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    self.draw_now_playing(ui);
                    ui.add_space(16.0);

                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("VISUALIZATION")
                                .color(self.current_palette.text_muted)
                                .monospace()
                                .size(11.0),
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if draw_hardware_button(ui, "\u{25B6}", &self.current_palette).clicked()
                            {
                                self.style = self.style.next();
                                self.refresh_palette();
                            }
                            ui.add_space(8.0);
                            ui.label(
                                egui::RichText::new(self.style.label())
                                    .color(self.current_palette.text_accent)
                                    .monospace()
                                    .size(13.0)
                                    .strong(),
                            );
                            ui.add_space(8.0);
                            if draw_hardware_button(ui, "\u{25C0}", &self.current_palette).clicked()
                            {
                                self.style = self.style.prev();
                                self.refresh_palette();
                            }
                        });
                    });

                    ui.add_space(8.0);
                    let remaining_height = ui.available_height();
                    self.draw_spectrum(ui, remaining_height);
                });
        });

        ui.ctx().request_repaint();
    }
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
fn main() {
    let (tx_dsp, rx_dsp) = mpsc::channel::<DspFrame>();
    let (tx_mpris, rx_mpris) = mpsc::channel::<MprisMessage>();

    let stream = spawn_audio_pipeline(tx_dsp);
    spawn_mpris_thread(tx_mpris);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([860.0, 520.0])
            .with_min_inner_size([400.0, 340.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Retro Audio Visualizer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());

            let has_wgpu = if let Some(rs) = &cc.wgpu_render_state {
                rs.renderer
                    .write()
                    .callback_resources
                    .insert(GpuVisualizerPipeline::new(&rs.device, rs.target_format));
                true
            } else {
                false
            };

            Ok(Box::new(VisualizerApp {
                rx_dsp,
                rx_mpris,
                latest_dsp: DspFrame::default(),
                smoothed_bars: [0.0; NUM_MEL_BARS],
                peak_bars: [0.0; NUM_MEL_BARS],
                screen_flash: 0.0,
                camera_shake: 0.0,
                album_art: None,
                title: String::new(),
                artist: String::new(),
                album: String::new(),
                record_angle: 0.0,
                pulse_rotation: 0.0,
                text_scroll: 0.0,
                is_playing: false,
                position: Duration::default(),
                length: None,
                volume: None,
                current_palette: VisualizerStyle::LedEqualizer.build_palette(None),
                style: VisualizerStyle::LedEqualizer,
                album_tint: None,
                has_wgpu,
                start_time: Instant::now(),
            }))
        }),
    )
    .unwrap();

    drop(stream);
}
