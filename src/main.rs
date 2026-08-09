use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use egui::ecolor::Hsva;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use egui::{
    epaint::Mesh, pos2, vec2, Align2, Color32, CornerRadius, FontId, Rect, Shape, Stroke,
    TextureHandle, TextureOptions,
};
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

const NUM_BARS: usize = 32;
const FFT_SIZE: usize = 1024;
const MIN_BIN: usize = 5;
const MAX_BIN: usize = 400;
const MIN_DB: f32 = -60.0;
const MAX_DB: f32 = 0.0;

#[cfg(target_os = "linux")]
const ALBUM_ART_SIZE: u32 = 160;

const MIN_BAR_WIDTH: f32 = 4.0;
const BAR_GAP: f32 = 3.0;
const MIN_SPECTRUM_HEIGHT: f32 = 80.0;

const ATTACK_RATE: f32 = 45.0; // ~0.07s to catch a rising value[cite: 1]
const RELEASE_RATE: f32 = 16.0; // ~0.19s to ease back down[cite: 1]
const PEAK_RELEASE_RATE: f32 = 2.2; // ~1.4s peak hold decay[cite: 1]

type BarFrame = [f32; NUM_BARS];

struct TrackUpdate {
    title: String,
    artist: String,
    album: String,
    art: Option<egui::ColorImage>,
}

struct PlaybackUpdate {
    is_playing: bool,
    position: Duration,
    length: Option<Duration>,
}

enum MprisMessage {
    Track(TrackUpdate),
    Playback(PlaybackUpdate),
}

// Add this anywhere above your VisualizerApp struct:
#[derive(Clone, Copy)]
struct AppPalette {
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
        // Your Cool Blue & Silver theme
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

impl AppPalette {
    fn from_image(img: &egui::ColorImage) -> Self {
        let mut r_sum = 0.0; let mut g_sum = 0.0; let mut b_sum = 0.0; let mut count = 0.0;
        let mut r_fall = 0.0; let mut g_fall = 0.0; let mut b_fall = 0.0; let mut fall_count = 0.0;

        // Sample the image pixels to find a dominant color
        for p in &img.pixels {
            let r = p.r() as f32; let g = p.g() as f32; let b = p.b() as f32;
            let hsva = Hsva::from(*p);

            // Prioritize colors that are reasonably saturated and not too dark/light
            if hsva.v > 0.2 && hsva.v < 0.8 && hsva.s > 0.3 {
                r_sum += r; g_sum += g; b_sum += b; count += 1.0;
            }
            r_fall += r; g_fall += g; b_fall += b; fall_count += 1.0;
        }

        let avg_color = if count > 0.0 {
            Color32::from_rgb((r_sum / count) as u8, (g_sum / count) as u8, (b_sum / count) as u8)
        } else if fall_count > 0.0 {
            Color32::from_rgb((r_fall / fall_count) as u8, (g_fall / fall_count) as u8, (b_fall / fall_count) as u8)
        } else {
            return Self::default();
        };

        let base_hsva = Hsva::from(avg_color);
        let h = base_hsva.h;
        let s = base_hsva.s.max(0.15); // Ensure at least a slight tint

        // Mathematically derive the UI colors from the dominant hue
        Self {
            bg_top: Hsva::new(h, s * 0.3, 0.22, 1.0).into(),
            bg_bottom: Hsva::new(h, s * 0.3, 0.12, 1.0).into(),
            panel_bg: Hsva::new(h, s * 0.2, 0.16, 1.0).into(),
            panel_border: Hsva::new(h, s * 0.15, 0.45, 1.0).into(),
            button_rest: Hsva::new(h, s * 0.2, 0.22, 1.0).into(),
            button_hover: Hsva::new(h, s * 0.2, 0.30, 1.0).into(),
            button_click: Hsva::new(h, s * 0.2, 0.15, 1.0).into(),
            text_muted: Hsva::new(h, s * 0.1, 0.65, 1.0).into(),
            text_primary: Hsva::new(h, s * 0.05, 0.95, 1.0).into(),
            text_accent: Hsva::new(h, (s * 1.5).min(1.0), 0.75, 1.0).into(),
            led_low: Hsva::new(h, s, 0.4, 1.0).into(),
            led_mid: Hsva::new(h, (s * 1.5).min(1.0), 0.75, 1.0).into(),
            led_high: Hsva::new(h, s * 0.5, 0.95, 1.0).into(),
            record_label: Hsva::new(h, s, 0.3, 1.0).into(),
        }
    }
}

// Helper for the progress bar text
fn format_time(d: Duration) -> String {
    let secs = d.as_secs();
    format!("{}:{:02}", secs / 60, secs % 60)
}

fn bin_to_bar(bin: usize) -> usize {
    let log_min = (MIN_BIN as f32).ln();
    let log_max = (MAX_BIN as f32).ln();
    let log_bin = (bin.clamp(MIN_BIN, MAX_BIN) as f32).ln();
    let ratio = (log_bin - log_min) / (log_max - log_min);
    ((ratio * NUM_BARS as f32) as usize).min(NUM_BARS - 1)
}

fn magnitude_to_normalized(magnitude: f32) -> f32 {
    let normalized_mag = magnitude / (FFT_SIZE as f32 * 0.5);
    let db = 20.0 * normalized_mag.max(1e-9).log10();
    ((db - MIN_DB) / (MAX_DB - MIN_DB)).clamp(0.0, 1.0)
}

fn compute_bars(spectrum: &[Complex<f32>]) -> BarFrame {
    let mut bars = [0.0f32; NUM_BARS];
    let max_bin = MAX_BIN.min(spectrum.len() - 1);
    for i in MIN_BIN..=max_bin {
        let normalized = magnitude_to_normalized(spectrum[i].norm());
        let bar = bin_to_bar(i);
        bars[bar] = bars[bar].max(normalized);
    }
    bars
}

fn exponential_smooth(current: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let alpha = 1.0 - (-rate * dt).exp();
    current + (target - current) * alpha
}

fn update_bars(current: &mut BarFrame, target: &BarFrame, attack_rate: f32, release_rate: f32, dt: f32) {
    for (c, &t) in current.iter_mut().zip(target.iter()) {
        let rate = if t > *c { attack_rate } else { release_rate };
        *c = exponential_smooth(*c, t, rate, dt);
    }
}

fn downsample_bars(bars: &BarFrame, display_count: usize) -> Vec<f32> {
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

fn spawn_audio_pipeline(tx_gui: mpsc::Sender<BarFrame>) -> cpal::Stream {
    let host = cpal::default_host();

    // On Windows, capture the Output device (Speakers) to hear desktop audio via loopback
    #[cfg(target_os = "windows")]
    let (device, config) = {
        let dev = host.default_output_device().expect("No default output device found!");
        let cfg = dev.default_output_config().expect("No default config found!");
        (dev, cfg)
    };

    // On Linux, capture the default Input device (Monitor)
    #[cfg(not(target_os = "windows"))]
    let (device, config) = {
        let dev = host.default_input_device().expect("No default input device found!");
        let cfg = dev.default_input_config().expect("No default config found!");
        (dev, cfg)
    };

    // IMPORTANT (Windows fix): WASAPI loopback capture on Windows delivers
    // stereo, interleaved buffers whose size depends on the audio engine's
    // period -- it is essentially never exactly FFT_SIZE samples, and it is
    // never mono. The old code assumed a mono callback of exactly FFT_SIZE
    // samples every time, which is true on the Linux monitor-source path but
    // false on Windows -- so `samples.len() != FFT_SIZE` was always true and
    // every buffer was silently dropped, meaning the FFT never ran and the
    // spectrum never moved. We downmix to mono here and accumulate into a
    // rolling buffer so a full FFT window is assembled regardless of the
    // device's native callback size or channel count.
    let channels = config.channels() as usize;
    let (tx_audio, rx_audio) = mpsc::channel::<Vec<f32>>();

    let stream = device
        .build_input_stream(
            config.config(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = if channels <= 1 {
                    data.to_vec()
                } else {
                    data.chunks_exact(channels)
                        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
                        .collect()
                };
                tx_audio.send(mono).ok();
            },
            |err| eprintln!("Audio stream error: {}", err),
            None,
        )
        .expect("Failed to build input stream!");

    stream.play().unwrap();

    thread::spawn(move || {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let mut input_vec = fft.make_input_vec();
        let mut output_vec = fft.make_output_vec();

        // Rolling accumulator: works no matter how big/small each device
        // callback's buffer is (fixes Windows; also makes Linux more robust).
        let mut acc: Vec<f32> = Vec::with_capacity(FFT_SIZE * 2);

        for samples in rx_audio {
            acc.extend_from_slice(&samples);

            while acc.len() >= FFT_SIZE {
                input_vec.copy_from_slice(&acc[..FFT_SIZE]);
                acc.drain(..FFT_SIZE);

                if fft.process(&mut input_vec, &mut output_vec).is_err() {
                    continue;
                }
                let bars = compute_bars(&output_vec);
                if tx_gui.send(bars).is_err() {
                    return;
                }
            }
        }
    });

    stream
}

#[cfg(target_os = "linux")]
fn load_album_art(uri: &str) -> Option<egui::ColorImage> {
    let img = if let Some(mut path_str) = uri.strip_prefix("file://") {
        if path_str.starts_with("//") {
            path_str = &path_str[1..];
        }
        if path_str.starts_with("localhost/") {
            path_str = &path_str[10..];
        }
        let clean_path = urlencoding::decode(path_str).unwrap_or_else(|_| path_str.into()).into_owned();
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

// --- LINUX MPRIS IMPLEMENTATION ---
//
// mpris::PlayerFinder::find_active() is explicitly documented as "naive": it
// does NOT check playback status, it just returns whichever player happens
// to be enumerated first over D-Bus. If Spotify is open but stopped/paused
// while Audacious (or anything else) is actually playing, find_active() can
// still hand back Spotify -- which is exactly why title/artist/art went
// blank and "nothing playing" showed up even though the spectrum was moving
// fine (spectrum capture is system-wide loopback and doesn't go through
// MPRIS at all, so it was never affected). We pick the player ourselves:
// prefer one that's actually Playing, fall back to one that's Paused, and
// otherwise treat it as nothing playing.
#[cfg(target_os = "linux")]
fn find_best_player(finder: &mpris::PlayerFinder) -> Option<mpris::Player<'_>> {
    let mut players = finder.find_all().ok()?;
    if players.is_empty() {
        return None;
    }

    let playing_idx = players.iter().position(|p| {
        matches!(p.get_playback_status(), Ok(mpris::PlaybackStatus::Playing))
    });
    let paused_idx = players.iter().position(|p| {
        matches!(p.get_playback_status(), Ok(mpris::PlaybackStatus::Paused))
    });

    playing_idx.or(paused_idx).map(|idx| players.swap_remove(idx))
}

#[cfg(target_os = "linux")]
fn spawn_mpris_thread(tx_mpris: mpsc::Sender<MprisMessage>) {
    thread::spawn(move || {
        let mut last_id: Option<String> = None;
        if let Ok(finder) = mpris::PlayerFinder::new() {
            loop {
                if let Some(player) = find_best_player(&finder) {
                    let status = player.get_playback_status().unwrap_or(mpris::PlaybackStatus::Stopped);
                    let is_playing = status == mpris::PlaybackStatus::Playing;
                    let position = player.get_position().unwrap_or_default();
                    
                    let metadata = player.get_metadata().ok();
                    let length = metadata.as_ref().and_then(|m| m.length());
                    
                    tx_mpris.send(MprisMessage::Playback(PlaybackUpdate {
                        is_playing, position, length
                    })).ok();

                    if let Some(metadata) = metadata {
                        let id = metadata.track_id().map(|id| id.to_string()).unwrap_or_else(|| {
                            format!("{:?}{:?}", metadata.title(), metadata.artists())
                        });

                        if last_id.as_deref() != Some(id.as_str()) {
                            last_id = Some(id);
                            let title = metadata.title().unwrap_or("Unknown title").to_string();
                            let artist = metadata.artists().map(|a| a.join(", ")).unwrap_or_else(|| "Unknown artist".to_string());
                            let album = metadata.album_name().unwrap_or("").to_string();
                            let art = metadata.art_url().and_then(load_album_art);

                            tx_mpris.send(MprisMessage::Track(TrackUpdate {
                                title, artist, album, art
                            })).ok();
                        }
                    }
                } else if last_id.is_some() {
                    // Nothing is Playing or Paused anymore -- clear stale info
                    // instead of leaving the last track/art stuck on screen.
                    last_id = None;
                    tx_mpris.send(MprisMessage::Track(TrackUpdate {
                        title: String::new(),
                        artist: String::new(),
                        album: String::new(),
                        art: None,
                    })).ok();
                    tx_mpris.send(MprisMessage::Playback(PlaybackUpdate {
                        is_playing: false,
                        position: Duration::default(),
                        length: None,
                    })).ok();
                }
                thread::sleep(Duration::from_millis(100)); 
            }
        }
    });
}

// --- WINDOWS IMPLEMENTATION (GSMTC, the Windows equivalent of MPRIS) ---
// Requires the `windows` crate with the "Media_Control", "Storage_Streams",
// and "Foundation" features enabled for the windows target -- see the
// Cargo.toml notes that go with this file. Verify feature names for your
// exact `windows` crate version at https://microsoft.github.io/windows-rs/features/
//
// NOTE: GSMTC only knows about apps that register with Windows' System Media
// Transport Controls (SMTC). Many Windows media apps (Spotify, foobar2000,
// AIMP, Chrome/Edge tabs, Windows Media Player, etc.) do this automatically,
// but plenty of others -- including stock Audacious on Windows -- do not, and
// won't appear here at all no matter what this code does. If a particular
// player never shows up, check whether it (or a plugin for it) supports SMTC.
//
// Also, like MPRIS's `find_active()`, GetCurrentSession() just returns
// whichever session Windows currently considers "the" session -- not
// necessarily whichever one is actually playing. We enumerate all sessions
// ourselves and prefer one that's actually Playing (falling back to Paused)
// instead, so an idle/stopped app can't shadow one that's really playing.
#[cfg(target_os = "linux")]
fn windows_best_session(
) -> Option<windows::Media::Control::GlobalSystemMediaTransportControlsSession> {
    use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager as SessionManager;
    use windows::Media::Control::GlobalSystemMediaTransportControlsSessionPlaybackStatus as PlaybackStatus;

    let manager = SessionManager::RequestAsync().ok()?.get().ok()?;
    let sessions = manager.GetSessions().ok()?;
    let count = sessions.Size().ok()?;

    let mut playing = None;
    let mut paused = None;

    for i in 0..count {
        let Ok(session) = sessions.GetAt(i) else {
            continue;
        };
        let status = session
            .GetPlaybackInfo()
            .ok()
            .and_then(|info| info.PlaybackStatus().ok());

        match status {
            Some(PlaybackStatus::Playing) if playing.is_none() => playing = Some(session),
            Some(PlaybackStatus::Paused) if paused.is_none() => paused = Some(session),
            _ => {}
        }
    }

    playing.or(paused)
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
                // 1. Get Playback Status & Position
                if let Ok(playback) = session.GetPlaybackInfo() {
                    let status = playback.PlaybackStatus().unwrap_or_default().0;
                    let is_playing = status == 4; // 4 = Playing

                    let mut position = Duration::default();
                    let mut length = None;

                    if let Ok(timeline) = session.GetTimelineProperties() {
                        let position_ticks = timeline.Position().unwrap_or_default().Duration as u64;
                        let last_updated = timeline.LastUpdatedTime().unwrap_or_default().UniversalTime as u64;
                        
                        // Convert Rust's SystemTime to Windows FILETIME (100-ns ticks since January 1, 1601)
                        let unix_now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_micros() as u64 * 10;
                        let win_now = unix_now + 116_444_736_000_000_000;
                        
                        // Add the time elapsed since the last Windows API update (only if playing!)
                        let time_elapsed = if is_playing && win_now > last_updated {
                            win_now - last_updated
                        } else {
                            0
                        };
                        
                        position = Duration::from_micros((position_ticks + time_elapsed) / 10);
                        length = Some(Duration::from_micros((timeline.EndTime().unwrap_or_default().Duration / 10) as u64));
                    }

                    tx_mpris.send(MprisMessage::Playback(PlaybackUpdate { is_playing, position, length })).ok();
                }

                // 2. Get Track Metadata & Album Art
                if let Ok(media_props) = session.TryGetMediaPropertiesAsync().and_then(|op| op.get()) {
                    let title = media_props.Title().unwrap_or_default().to_string();
                    let artist = media_props.Artist().unwrap_or_default().to_string();
                    let album = media_props.AlbumTitle().unwrap_or_default().to_string();

                    let track_id = format!("{}-{}-{}", title, artist, album);
                    if track_id != last_id {
                        last_id = track_id;
                        let mut art = None;
                        
                        // Extract Album Art from Windows Memory Stream
                        if let Ok(thumb_ref) = media_props.Thumbnail() {
                            if let Ok(stream) = thumb_ref.OpenReadAsync().and_then(|op| op.get()) {
                                let size = stream.Size().unwrap_or(0) as u32;
                                if size > 0 {
                                    if let Ok(reader) = DataReader::CreateDataReader(&stream) {
                                        if reader.LoadAsync(size).and_then(|op| op.get()).is_ok() {
                                            let mut bytes = vec![0u8; size as usize];
                                            if reader.ReadBytes(&mut bytes).is_ok() {
                                                if let Ok(img) = image::load_from_memory(&bytes) {
                                                    let img = img.resize_to_fill(160, 160, image::imageops::FilterType::Lanczos3);
                                                    let rgba = img.to_rgba8();
                                                    art = Some(egui::ColorImage::from_rgba_unmultiplied(
                                                        [rgba.width() as usize, rgba.height() as usize],
                                                        &rgba.into_raw(),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        tx_mpris.send(MprisMessage::Track(TrackUpdate { title, artist, album, art })).ok();
                    }
                }
            } else {
                tx_mpris.send(MprisMessage::Playback(PlaybackUpdate { is_playing: false, position: Duration::default(), length: None })).ok();
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

struct VisualizerApp {
    rx_gui: mpsc::Receiver<BarFrame>,
    rx_mpris: mpsc::Receiver<MprisMessage>, // Renamed from rx_meta
    current_bars: BarFrame,
    peak_bars: BarFrame,
    attack_rate: f32,
    release_rate: f32,
    peak_release_rate: f32,
    album_art: Option<TextureHandle>,
    title: String,
    artist: String,
    album: String,
    record_angle: f32,
    text_scroll: f32,
    is_playing: bool,            // NEW
    position: Duration,          // NEW
    length: Option<Duration>,    // NEW
    current_palette: AppPalette,
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
    painter.rect_stroke(rect, CornerRadius::same(3), Stroke::new(1.0, palette.panel_border), egui::StrokeKind::Inside);

    let text_color = if is_clicked { palette.text_primary } else { palette.text_accent };

    painter.text(rect.center(), Align2::CENTER_CENTER, text, FontId::proportional(16.0), text_color);
    response
}

fn draw_marquee_text(ui: &mut egui::Ui, text: &str, font: FontId, color: Color32, offset: f32, max_width: f32) {
    // FIX: Lay out the text first without holding onto `painter`
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, color);
    let text_width = galley.rect.width();

    // Safely allocate size (mutable borrow)
    let (rect, _) = ui.allocate_exact_size(vec2(max_width, galley.rect.height()), egui::Sense::hover());

    // Grab the painter (immutable borrow) now that we're done mutating `ui`
    let painter = ui.painter();

    if text_width <= max_width {
        painter.galley(rect.left_top(), galley, color);
    } else {
        let scroll_max = text_width + 40.0;
        let current_offset = offset % scroll_max;

        painter.with_clip_rect(rect).galley(rect.left_top() - vec2(current_offset, 0.0), galley.clone(), color);

        if current_offset > (text_width + 40.0 - max_width) {
            painter.with_clip_rect(rect).galley(rect.left_top() + vec2(scroll_max - current_offset, 0.0), galley, color);
        }
    }
}

impl VisualizerApp {
    fn draw_now_playing(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // -- The Sleeve and Spinning Vinyl --
            let sleeve_size = 140.0;
            let record_radius = sleeve_size / 2.0;
            let protruding_amount = record_radius * 0.85;

            let (rect, _) = ui.allocate_exact_size(vec2(sleeve_size + protruding_amount, sleeve_size), egui::Sense::hover());
            let painter = ui.painter();

            let sleeve_rect = Rect::from_min_size(rect.left_top(), vec2(sleeve_size, sleeve_size));
            let record_center = pos2(sleeve_rect.right() - record_radius + protruding_amount, sleeve_rect.center().y);

            // 1. Draw the Spinning Vinyl (underneath the sleeve)
            painter.circle_filled(record_center, record_radius, Color32::from_rgb(18, 18, 18));
            painter.circle_stroke(record_center, record_radius * 0.85, Stroke::new(1.0, Color32::from_rgb(35, 35, 35)));
            painter.circle_stroke(record_center, record_radius * 0.70, Stroke::new(1.0, Color32::from_rgb(25, 25, 25)));
            painter.circle_stroke(record_center, record_radius * 0.55, Stroke::new(1.0, Color32::from_rgb(35, 35, 35)));

            // Dynamic center label
            let label_radius = record_radius * 0.42;
            painter.circle_filled(record_center, label_radius, self.current_palette.record_label);

            let rot_dir = vec2(self.record_angle.cos(), self.record_angle.sin());
            let marker_color = self.current_palette.text_primary;

            // Inner ring
            painter.circle_stroke(record_center, label_radius * 0.8, Stroke::new(1.0, marker_color));

            // Markers
            let dot_pos = record_center + rot_dir * (label_radius * 0.5);
            painter.circle_filled(dot_pos, label_radius * 0.15, marker_color);

            let line_start = record_center - rot_dir * (label_radius * 0.8);
            let line_end = record_center - rot_dir * (label_radius * 0.3);
            painter.line_segment([line_start, line_end], Stroke::new(3.0, marker_color));

            // Spindle hole
            painter.circle_filled(record_center, 4.0, Color32::from_rgb(20, 20, 20));

            // 2. Draw the Album Sleeve (on top)
            painter.rect_filled(sleeve_rect, CornerRadius::same(4), self.current_palette.bg_bottom);

            if let Some(tex) = &self.album_art {
                painter.image(
                    tex.id(),
                              sleeve_rect,
                              Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)),
                              Color32::WHITE
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

            // Dynamic inner shadow/border
            painter.rect_stroke(sleeve_rect, CornerRadius::same(4), Stroke::new(1.0, self.current_palette.panel_border), egui::StrokeKind::Inside);

            ui.add_space(24.0);

            // -- Metadata & Controls --
            ui.vertical(|ui| {
                ui.add_space(16.0);

                ui.label(egui::RichText::new("NOW PLAYING")
                .size(11.0)
                .monospace()
                .color(self.current_palette.text_muted));

                let title = if self.title.is_empty() { "NOTHING PLAYING" } else { &self.title };
                let available_text_width = ui.available_width() - 20.0;

                draw_marquee_text(
                    ui,
                    &title.to_uppercase(),
                                  FontId::monospace(22.0),
                                  self.current_palette.text_primary,
                                  self.text_scroll,
                                  available_text_width
                );

                ui.add_space(4.0);

                if !self.artist.is_empty() {
                    draw_marquee_text(
                        ui,
                        &self.artist.to_uppercase(),
                                      FontId::monospace(14.0),
                                      self.current_palette.text_accent,
                                      self.text_scroll * 0.7,
                                      available_text_width
                    );
                }
                if !self.album.is_empty() {
                    draw_marquee_text(
                        ui,
                        &self.album.to_uppercase(),
                                      FontId::monospace(12.0),
                                      self.current_palette.text_muted,
                                      self.text_scroll * 0.5,
                                      available_text_width
                    );
                }

                ui.add_space(8.0);

                // --- Retro Progress Bar ---
                let progress_rect_height = 16.0;
                let (prog_rect, _) = ui.allocate_exact_size(vec2(available_text_width, progress_rect_height), egui::Sense::hover());
                let prog_painter = ui.painter(); // Safely grab a painter for the progress bar

                let time_str = format_time(self.position);
                let rem_str = if let Some(len) = self.length {
                    let rem = len.saturating_sub(self.position);
                    format!("-{}", format_time(rem))
                } else {
                    "--:--".to_string()
                };

                prog_painter.text(prog_rect.left_center(), Align2::LEFT_CENTER, time_str, FontId::monospace(11.0), self.current_palette.text_muted);
                prog_painter.text(prog_rect.right_center(), Align2::RIGHT_CENTER, rem_str, FontId::monospace(11.0), self.current_palette.text_muted);

                let bar_left = prog_rect.left() + 38.0;
                let bar_right = prog_rect.right() - 42.0;
                let bar_y = prog_rect.center().y;

                // Background Track
                prog_painter.line_segment([pos2(bar_left, bar_y), pos2(bar_right, bar_y)], Stroke::new(2.0, self.current_palette.button_rest));

                // Active Track & Playhead
                if let Some(len) = self.length {
                    if len.as_secs_f32() > 0.0 {
                        let ratio = (self.position.as_secs_f32() / len.as_secs_f32()).clamp(0.0, 1.0);
                        let playhead_x = bar_left + (bar_right - bar_left) * ratio;

                        prog_painter.line_segment([pos2(bar_left, bar_y), pos2(playhead_x, bar_y)], Stroke::new(2.0, self.current_palette.text_accent));
                        prog_painter.circle_filled(pos2(playhead_x, bar_y), 3.5, self.current_palette.text_primary);
                    }
                }

                ui.add_space(12.0);

                // -- Physical Custom Buttons --
                ui.horizontal(|ui| {
                    if draw_hardware_button(ui, "⏮", &self.current_palette).clicked() {
                        #[cfg(target_os = "linux")]
                        std::thread::spawn(|| { if let Ok(finder) = mpris::PlayerFinder::new() { if let Ok(player) = finder.find_active() { player.previous().ok(); } } });
                        
                        #[cfg(target_os = "windows")]
                        std::thread::spawn(|| {
                            use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                            if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                if let Ok(session) = manager.GetCurrentSession() { session.TrySkipPreviousAsync().ok(); }
                            }
                        });
                    }
                    if draw_hardware_button(ui, "⏵⏸", &self.current_palette).clicked() {
                        #[cfg(target_os = "linux")]
                        std::thread::spawn(|| { if let Ok(finder) = mpris::PlayerFinder::new() { if let Ok(player) = finder.find_active() { player.play_pause().ok(); } } });
                        
                        #[cfg(target_os = "windows")]
                        std::thread::spawn(|| {
                            use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                            if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                if let Ok(session) = manager.GetCurrentSession() { session.TryTogglePlayPauseAsync().ok(); }
                            }
                        });
                    }
                    if draw_hardware_button(ui, "⏭", &self.current_palette).clicked() {
                        #[cfg(target_os = "linux")]
                        std::thread::spawn(|| { if let Ok(finder) = mpris::PlayerFinder::new() { if let Ok(player) = finder.find_active() { player.next().ok(); } } });
                        
                        #[cfg(target_os = "windows")]
                        std::thread::spawn(|| {
                            use windows::Media::Control::GlobalSystemMediaTransportControlsSessionManager;
                            if let Ok(manager) = GlobalSystemMediaTransportControlsSessionManager::RequestAsync().and_then(|op| op.get()) {
                                if let Ok(session) = manager.GetCurrentSession() { session.TrySkipNextAsync().ok(); }
                            }
                        });
                    }
                });
            });
        });
    }

    fn draw_spectrum(&self, ui: &mut egui::Ui, height: f32) {
        // These were the missing setup lines!
        let desired_size = vec2(ui.available_width(), height.max(MIN_SPECTRUM_HEIGHT));
        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
        let painter = ui.painter();

        let available_width = rect.width().max(1.0);
        let max_bars_that_fit = (((available_width + BAR_GAP) / (MIN_BAR_WIDTH + BAR_GAP)).floor() as usize).max(1);
        let n = max_bars_that_fit.min(self.current_bars.len());

        let values = downsample_bars(&self.current_bars, n);
        let peaks = downsample_bars(&self.peak_bars, n);

        let bar_width = ((available_width - BAR_GAP * (n as f32 - 1.0)) / n as f32).max(1.0);

        // --- Dynamic Vertical Scaling ---
        let segment_gap = 2.0;
        let desired_segment_height = (bar_width * 0.5).clamp(4.0, 12.0);
        let segments = ((rect.height() + segment_gap) / (desired_segment_height + segment_gap)).floor() as usize;
        let segments = segments.clamp(6, 64);
        let segment_height = (rect.height() - (segments - 1) as f32 * segment_gap) / segments as f32;

        for (i, (&value, &peak)) in values.iter().zip(peaks.iter()).enumerate() {
            let x = rect.left() + i as f32 * (bar_width + BAR_GAP);

            let active_segments = (value * segments as f32).round() as usize;
            let peak_segment = (peak * segments as f32).round() as usize;

            for s in 0..segments {
                let y = rect.bottom() - (s as f32 + 1.0) * segment_height - s as f32 * segment_gap;
                let seg_rect = Rect::from_min_max(pos2(x, y), pos2(x + bar_width, y + segment_height));

                let is_active = s < active_segments;
                let is_peak = s == peak_segment;

                // Dynamically map colors based on the total number of segments
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
                    base_color.linear_multiply(0.12) // Dim inactive LED
                };

                painter.rect_filled(seg_rect, CornerRadius::same(1), color);

                if is_active {
                    painter.rect_filled(seg_rect.expand(2.5), CornerRadius::same(2), with_alpha(base_color, 25));
                }
            }
        }
    }
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dt = ui.input(|i| i.stable_dt).clamp(1.0 / 240.0, 0.1);

        // Advance animations & SMOOTH PROGRESS BAR
        if self.is_playing {
            self.record_angle += dt * 1.8; 
            self.text_scroll += dt * 35.0; 
            self.position += Duration::from_secs_f32(dt); // Glide forward mathematically
        }

        let mut latest_bars = None;
        while let Ok(target_bars) = self.rx_gui.try_recv() {
            latest_bars = Some(target_bars);
        }
        if let Some(target_bars) = latest_bars {
            update_bars(&mut self.current_bars, &target_bars, self.attack_rate, self.release_rate, dt);
        }
        for (peak, &current) in self.peak_bars.iter_mut().zip(self.current_bars.iter()) {
            let decayed = *peak * (-self.peak_release_rate * dt).exp();
            *peak = current.max(decayed);
        }

        let mut latest_playback = None;
        while let Ok(msg) = self.rx_mpris.try_recv() {
            match msg {
                MprisMessage::Track(update) => {
                    self.title = update.title;
                    self.artist = update.artist;
                    self.album = update.album;
                    if let Some(img) = &update.art {
                        self.current_palette = AppPalette::from_image(img);
                    } else {
                        self.current_palette = AppPalette::default();
                    }
                    self.album_art = update.art.map(|color_image| {
                        ui.ctx().load_texture("album_art", color_image, TextureOptions::LINEAR)
                    });
                    self.text_scroll = 0.0; 
                }
                MprisMessage::Playback(update) => {
                    latest_playback = Some(update);
                }
            }
        }

        // Apply thread updates, but prevent backward UI stutters
        if let Some(update) = latest_playback {
            self.is_playing = update.is_playing;
            self.length = update.length;
            
            // Allow the UI to glide smoothly, but if it drifts away from the accurate 
            // background thread time by more than 1.5 seconds, snap it to the exact time.
            let diff = if self.position > update.position {
                self.position - update.position
            } else {
                update.position - self.position
            };
            
            if diff.as_secs_f32() > 1.5 {
                self.position = update.position;
            }
        }

        paint_dynamic_gradient_background(ui.painter(), ui.max_rect(), &self.current_palette);

        let outer_margin = egui::Margin::same(24);
        let background = egui::Frame::default().inner_margin(outer_margin);

        background.show(ui, |ui| {
            // Draw brushed inner panel container (Steely gray)
            let panel_rect = ui.available_rect_before_wrap();
            ui.painter().rect_filled(panel_rect, CornerRadius::same(6), self.current_palette.panel_bg);
            ui.painter().rect_stroke(panel_rect, CornerRadius::same(6), Stroke::new(1.0, self.current_palette.panel_border), egui::StrokeKind::Inside);

            // Inner margin for content inside the metal panel
            egui::Frame::default().inner_margin(egui::Margin::same(20)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("STEREO VISUALIZER").color(self.current_palette.text_primary).monospace().size(16.0).strong());

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // Power Light LED (Cyan)
                        let (rect, _) = ui.allocate_exact_size(vec2(12.0, 12.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 4.0, self.current_palette.text_accent);
                        ui.painter().circle_filled(rect.center(), 6.0, with_alpha(self.current_palette.text_accent, 60));
                        ui.label(egui::RichText::new("POWER").monospace().size(10.0).color(self.current_palette.text_muted));
                    });
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                self.draw_now_playing(ui);

                ui.add_space(20.0);

                ui.label(egui::RichText::new("GRAPHIC EQUALIZER")
                .color(Color32::from_rgb(150, 160, 175))
                .monospace()
                .size(12.0));
                ui.add_space(8.0);

                let remaining_height = ui.available_height();
                self.draw_spectrum(ui, remaining_height);
            });
        });

        ui.ctx().request_repaint();
    }
}

fn main() {
    let (tx_gui, rx_gui) = mpsc::channel::<BarFrame>();
    let (tx_mpris, rx_mpris) = mpsc::channel::<MprisMessage>(); // Using new message Enum

    let stream = spawn_audio_pipeline(tx_gui);
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
            Ok(Box::new(VisualizerApp {
                rx_gui,
                rx_mpris, // Renamed
                current_bars: [0.0; NUM_BARS],
                peak_bars: [0.0; NUM_BARS],
                attack_rate: ATTACK_RATE,
                release_rate: RELEASE_RATE,
                peak_release_rate: PEAK_RELEASE_RATE,
                album_art: None,
                current_palette: AppPalette::default(),
                title: String::new(),
                        artist: String::new(),
                        album: String::new(),
                        record_angle: 0.0,
                        text_scroll: 0.0,
                        is_playing: false,                   // NEW
                        position: Duration::default(),       // NEW
                        length: None,                        // NEW
            }))
        }),
    )
    .unwrap();

    drop(stream);
}
