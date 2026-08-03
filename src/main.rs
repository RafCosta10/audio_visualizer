use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use egui::ecolor::Hsva;
use egui::{
    epaint::Mesh, pos2, vec2, Align2, Color32, CornerRadius, FontId, Image, Rect, Shape, Stroke,
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
const ALBUM_ART_SIZE: u32 = 160;

// Bars never get thinner than this - below this width, adjacent bars are
// merged (max-pooled) so the spectrum degrades to fewer, chunkier bars
// instead of vanishing.
const MIN_BAR_WIDTH: f32 = 4.0;
const BAR_GAP: f32 = 3.0;
const MIN_SPECTRUM_HEIGHT: f32 = 60.0;

// Smoothing rates, in units of "per second" so motion looks the same
// regardless of frame rate. Attack is fast (bars jump up to catch transients),
// release is slightly slower (bars ease down instead of snapping), and the
// peak markers fall slower still so they're readable as a hold indicator.
// Rule of thumb: time to reach ~95% of a new value ≈ 3 / rate (seconds).
const ATTACK_RATE: f32 = 45.0; // ~0.07s to catch a rising value
const RELEASE_RATE: f32 = 16.0; // ~0.19s to ease back down
const PEAK_RELEASE_RATE: f32 = 2.2; // ~1.4s peak hold decay

type BarFrame = [f32; NUM_BARS];

struct TrackUpdate {
    title: String,
    artist: String,
    album: String,
    art: Option<egui::ColorImage>,
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

/// Exponential smoothing that behaves the same regardless of frame rate:
/// `rate` is a "per second" speed, so a slow frame and several fast frames
/// covering the same wall-clock time land on (almost) the same value.
fn exponential_smooth(current: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let alpha = 1.0 - (-rate * dt).exp();
    current + (target - current) * alpha
}

/// Move `current` toward `target`, using a fast attack when the signal is
/// rising and a slower release when it's falling. This is what makes the
/// bars feel punchy on transients but calm on decay, instead of snapping
/// instantly in both directions.
fn update_bars(current: &mut BarFrame, target: &BarFrame, attack_rate: f32, release_rate: f32, dt: f32) {
    for (c, &t) in current.iter_mut().zip(target.iter()) {
        let rate = if t > *c { attack_rate } else { release_rate };
        *c = exponential_smooth(*c, t, rate, dt);
    }
}

/// Max-pool the fixed NUM_BARS-wide FFT data down to `display_count` bars.
/// Using max (rather than average) keeps transients visible even when
/// several source bins get merged into one displayed bar.
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
    let device = host
    .default_input_device()
    .expect("No default input device found!");
    let config = device.default_input_config().expect("No config found!");

    let (tx_audio, rx_audio) = mpsc::channel::<Vec<f32>>();

    let stream = device
    .build_input_stream(
        config.config(),
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            tx_audio.send(data.to_vec()).ok();
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

        for samples in rx_audio {
            if samples.len() != FFT_SIZE {
                continue;
            }
            input_vec.copy_from_slice(&samples);
            if fft.process(&mut input_vec, &mut output_vec).is_err() {
                continue;
            }
            let bars = compute_bars(&output_vec);
            if tx_gui.send(bars).is_err() {
                break;
            }
        }
    });

    stream
}

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

fn spawn_mpris_thread(tx_meta: mpsc::Sender<TrackUpdate>) {
    thread::spawn(move || {
        let mut last_id: Option<String> = None;
        loop {
            if let Ok(finder) = mpris::PlayerFinder::new() {
                if let Ok(player) = finder.find_active() {
                    if let Ok(metadata) = player.get_metadata() {
                        let id = metadata
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

                            tx_meta
                            .send(TrackUpdate {
                                title,
                                artist,
                                album,
                                art,
                            })
                            .ok();
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(800));
        }
    });
}

struct VisualizerApp {
    rx_gui: mpsc::Receiver<BarFrame>,
    rx_meta: mpsc::Receiver<TrackUpdate>,
    current_bars: BarFrame,
    peak_bars: BarFrame,
    attack_rate: f32,
    release_rate: f32,
    peak_release_rate: f32,
    album_art: Option<TextureHandle>,
    title: String,
    artist: String,
    album: String,
}

fn with_alpha(color: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha)
}

fn paint_gradient_background(painter: &egui::Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 2, 3);
    painter.add(Shape::mesh(mesh));
}

impl VisualizerApp {
    fn draw_now_playing(&self, ui: &mut egui::Ui) {
        // Scale the art and type down at narrow widths instead of letting
        // them overflow the window or get clipped.
        let available_width = ui.available_width();
        let compact = available_width < 380.0;
        let art_px = if compact { 64.0 } else { 96.0 };
        let title_size = if compact { 18.0 } else { 22.0 };

        ui.horizontal(|ui| {
            let art_size = vec2(art_px, art_px);
            if let Some(tex) = &self.album_art {
                ui.add(Image::new(tex).fit_to_exact_size(art_size));
            } else {
                let (rect, _) = ui.allocate_exact_size(art_size, egui::Sense::hover());
                ui.painter()
                .rect_filled(rect, CornerRadius::same(10), Color32::from_rgb(38, 38, 52));
                ui.painter().text(
                    rect.center(),
                                  Align2::CENTER_CENTER,
                                  "\u{266A}",
                                  FontId::proportional(art_px * 0.35),
                                  Color32::from_rgb(120, 120, 150),
                );
            }

            ui.add_space(if compact { 10.0 } else { 16.0 });
            ui.vertical(|ui| {
                ui.add_space(6.0);
                let title = if self.title.is_empty() {
                    "Nothing playing"
                } else {
                    &self.title
                };
                // truncate() keeps long metadata from a track's title/artist/
                // album pushing the window wider or overflowing the layout;
                // it ellipsizes instead.
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(title)
                        .size(title_size)
                        .strong()
                        .color(Color32::WHITE),
                    )
                    .truncate(),
                );
                if !self.artist.is_empty() {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&self.artist)
                            .size(15.0)
                            .color(Color32::from_rgb(170, 170, 190)),
                        )
                        .truncate(),
                    );
                }
                if !self.album.is_empty() && !compact {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&self.album)
                            .size(13.0)
                            .italics()
                            .color(Color32::from_rgb(120, 120, 145)),
                        )
                        .truncate(),
                    );
                }

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("⏮").clicked() {
                        thread::spawn(|| {
                            if let Ok(finder) = mpris::PlayerFinder::new() {
                                if let Ok(player) = finder.find_active() {
                                    player.previous().ok();
                                }
                            }
                        });
                    }
                    if ui.button("⏵⏸").clicked() {
                        thread::spawn(|| {
                            if let Ok(finder) = mpris::PlayerFinder::new() {
                                if let Ok(player) = finder.find_active() {
                                    player.play_pause().ok();
                                }
                            }
                        });
                    }
                    if ui.button("⏭").clicked() {
                        thread::spawn(|| {
                            if let Ok(finder) = mpris::PlayerFinder::new() {
                                if let Ok(player) = finder.find_active() {
                                    player.next().ok();
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
        let painter = ui.painter();

        // Figure out how many bars actually fit at a legible width, and
        // merge the full-resolution FFT data down to that count. This is
        // what keeps bars visible (instead of shrinking to nothing) as the
        // window gets narrower - fewer, wider bars rather than 32 hairlines.
        let available_width = rect.width().max(1.0);
        let max_bars_that_fit =
        (((available_width + BAR_GAP) / (MIN_BAR_WIDTH + BAR_GAP)).floor() as usize).max(1);
        let n = max_bars_that_fit.min(self.current_bars.len());

        let values = downsample_bars(&self.current_bars, n);
        let peaks = downsample_bars(&self.peak_bars, n);

        let gap = BAR_GAP;
        let bar_width = ((available_width - gap * (n as f32 - 1.0)) / n as f32).max(1.0);
        let baseline = rect.bottom();

        for (i, (&value, &peak)) in values
            .iter()
            .zip(peaks.iter())
            .enumerate()
            {
                let x = rect.left() + i as f32 * (bar_width + gap);
                let height = (value * rect.height()).clamp(0.0, rect.height());
                let bar_rect =
                Rect::from_min_max(pos2(x, baseline - height), pos2(x + bar_width, baseline));

                let hue = 0.55 + (i as f32 / n as f32) * 0.30;
                let brightness = 0.45 + value.clamp(0.0, 1.0) * 0.55;
                let color: Color32 = Hsva::new(hue, 0.75, brightness, 1.0).into();

                for (expand, alpha) in [(8.0, 10u8), (4.0, 18u8)] {
                    let glow_rect = bar_rect.expand(expand);
                    painter.rect_filled(
                        glow_rect,
                        CornerRadius::same(((bar_width * 0.5 + expand) as u8).max(1)),
                                        with_alpha(color, alpha),
                    );
                }

                painter.rect_filled(
                    bar_rect,
                    CornerRadius::same(((bar_width * 0.4) as u8).max(1)),
                                    color,
                );

                let reflection_height = height * 0.3;
                if reflection_height > 1.0 {
                    let reflection_rect = Rect::from_min_max(
                        pos2(x, baseline),
                                                             pos2(x + bar_width, baseline + reflection_height),
                    );
                    painter.rect_filled(
                        reflection_rect,
                        CornerRadius::same(((bar_width * 0.3) as u8).max(1)),
                                        with_alpha(color, 40),
                    );
                }

                let peak_height = (peak * rect.height()).clamp(0.0, rect.height());
                let peak_y = baseline - peak_height;
                painter.line_segment(
                    [pos2(x, peak_y), pos2(x + bar_width, peak_y)],
                                     Stroke::new(2.0, Color32::WHITE.gamma_multiply(0.8)),
                );
            }
    }
}

impl eframe::App for VisualizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Real elapsed time since the last frame - repaint is requested
        // unconditionally below, so frame rate can vary a lot; smoothing by
        // dt instead of by a fixed per-frame factor keeps bar motion looking
        // the same regardless of how fast the display is repainting.
        let dt = ui.input(|i| i.stable_dt).clamp(1.0 / 240.0, 0.1);

        // The audio thread can push several FFT frames per GUI repaint;
        // only the newest one matters as this frame's target, so drain the
        // channel and smooth toward that once rather than once per message
        // (which would make bars decay faster whenever a backlog builds up).
        let mut latest_bars = None;
        while let Ok(target_bars) = self.rx_gui.try_recv() {
            latest_bars = Some(target_bars);
        }
        if let Some(target_bars) = latest_bars {
            update_bars(
                &mut self.current_bars,
                &target_bars,
                self.attack_rate,
                self.release_rate,
                dt,
            );
        }
        for (peak, &current) in self.peak_bars.iter_mut().zip(self.current_bars.iter()) {
            let decayed = *peak * (-self.peak_release_rate * dt).exp();
            *peak = current.max(decayed);
        }

        while let Ok(update) = self.rx_meta.try_recv() {
            self.title = update.title;
            self.artist = update.artist;
            self.album = update.album;
            self.album_art = update.art.map(|color_image| {
                ui.ctx().load_texture(
                    "album_art",
                    color_image,
                    TextureOptions::LINEAR,
                )
            });
        }

        // Subtle top-to-bottom gradient reads as less flat/static than a
        // solid fill, especially behind the bar glow effects.
        paint_gradient_background(
            ui.painter(),
                                  ui.max_rect(),
                                  Color32::from_rgb(16, 16, 26),
                                  Color32::from_rgb(9, 9, 15),
        );

        let background = egui::Frame::default().inner_margin(egui::Margin::same(24));

        background.show(ui, |ui| {
            ui.heading(egui::RichText::new("Audio Visualizer").color(Color32::WHITE));
            ui.add_space(16.0);
            self.draw_now_playing(ui);
            ui.add_space(24.0);
            // Whatever vertical space is left (rather than a hardcoded
            // height) so the spectrum actually grows/shrinks when the
            // window is resized taller or shorter.
            let remaining_height = ui.available_height();
            self.draw_spectrum(ui, remaining_height);
        });

        ui.ctx().request_repaint();
    }
}

fn main() {
    let (tx_gui, rx_gui) = mpsc::channel::<BarFrame>();
    let (tx_meta, rx_meta) = mpsc::channel::<TrackUpdate>();

    let stream = spawn_audio_pipeline(tx_gui);
    spawn_mpris_thread(tx_meta);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
        .with_inner_size([720.0, 480.0])
        // Keep enough room for the now-playing row and a sliver of
        // spectrum; the layout itself degrades gracefully below this too,
        // but there's no point letting the window shrink to nothing.
        .with_min_inner_size([220.0, 260.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Audio Visualizer",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(VisualizerApp {
                rx_gui,
                rx_meta,
                current_bars: [0.0; NUM_BARS],
                peak_bars: [0.0; NUM_BARS],
                attack_rate: ATTACK_RATE,
                release_rate: RELEASE_RATE,
                peak_release_rate: PEAK_RELEASE_RATE,
                album_art: None,
                title: String::new(),
                        artist: String::new(),
                        album: String::new(),
            }))
        }),
    )
    .unwrap();

    drop(stream);
}
