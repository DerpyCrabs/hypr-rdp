//! NVIDIA H.264 encoding through libavcodec, with an opaque, version-independent C bridge.
use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr::NonNull;

use anyhow::{bail, ensure, Context, Result};

use super::h264::extract_sps_pps;
use super::H264RateControl;

extern "C" {
    fn hypr_nvenc_new(
        width: c_int,
        height: c_int,
        bitrate: i64,
        fps: c_int,
        quality: c_int,
        cqp: c_int,
        error: *mut c_char,
        error_size: usize,
    ) -> *mut c_void;
    fn hypr_nvenc_free(encoder: *mut c_void);
    fn hypr_nvenc_encode(
        encoder: *mut c_void,
        bgra: *const u8,
        stride: c_int,
        force_idr: c_int,
        data: *mut *const u8,
        size: *mut c_int,
        error: *mut c_char,
        error_size: usize,
    ) -> c_int;
}

pub struct NvencEncoder {
    inner: NonNull<c_void>,
    width: usize,
    height: usize,
    force_idr: bool,
    cached_sps_pps: Option<Vec<u8>>,
}

// SAFETY: The context is exclusively owned. All FFmpeg operations require &mut self;
// moving it between threads never permits concurrent access to the codec or frame.
unsafe impl Send for NvencEncoder {}

impl Drop for NvencEncoder {
    fn drop(&mut self) {
        // SAFETY: inner is created by hypr_nvenc_new and freed exactly once.
        unsafe { hypr_nvenc_free(self.inner.as_ptr()) };
    }
}

impl NvencEncoder {
    pub fn new(
        width: u32,
        height: u32,
        bitrate: u32,
        fps: u32,
        quality: u8,
        rate_control: H264RateControl,
    ) -> Result<Self> {
        ensure!(
            width > 0 && height > 0 && width.is_multiple_of(2) && height.is_multiple_of(2),
            "NVENC AVC420 requires positive even dimensions"
        );
        let width_i32 = i32::try_from(width).context("NVENC width exceeds i32")?;
        let height_i32 = i32::try_from(height).context("NVENC height exceeds i32")?;
        ensure!(width_i32 <= i32::MAX / 4, "NVENC BGRA stride exceeds i32");
        ensure!(
            fps > 0 && fps <= (i32::MAX / 2) as u32,
            "invalid NVENC frame rate"
        );
        ensure!(quality <= 51, "NVENC quality must be in 0..=51");
        ensure!(
            bitrate > 0 || rate_control == H264RateControl::Cqp,
            "NVENC VBR bitrate must be positive"
        );
        let mut error = [0 as c_char; 512];
        // SAFETY: Arguments are validated above; error is a writable fixed-size buffer.
        let inner = unsafe {
            hypr_nvenc_new(
                width_i32,
                height_i32,
                i64::from(bitrate),
                fps as i32,
                i32::from(quality),
                i32::from(rate_control == H264RateControl::Cqp),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let inner = NonNull::new(inner).with_context(|| error_message(&error))?;
        Ok(Self {
            inner,
            width: width as usize,
            height: height as usize,
            force_idr: true,
            cached_sps_pps: None,
        })
    }

    pub fn encode(&mut self, bgra: &[u8], stride: usize) -> Result<Vec<u8>> {
        validate_input(self.width, self.height, bgra.len(), stride)?;
        let mut error = [0 as c_char; 512];
        let mut data = std::ptr::null();
        let mut size = 0;
        // SAFETY: validate_input checks the complete source footprint and stride.
        // The context is exclusively borrowed; returned packet memory lives until
        // the next encode/drop and is copied before either can occur.
        let result = unsafe {
            hypr_nvenc_encode(
                self.inner.as_ptr(),
                bgra.as_ptr(),
                stride as i32,
                i32::from(self.force_idr),
                &mut data,
                &mut size,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if result < 0 {
            bail!("{}", error_message(&error));
        }
        ensure!(
            !data.is_null() && size > 0,
            "NVENC returned an empty packet"
        );
        // SAFETY: Successful C bridge call returns size initialized bytes owned by AVPacket.
        let packet = unsafe { std::slice::from_raw_parts(data, size as usize) };
        self.force_idr = false;
        // Match the software backend's compatibility behavior for Windows MFT decoders.
        if let Some(headers) = extract_sps_pps(packet) {
            self.cached_sps_pps = Some(headers);
            Ok(packet.to_vec())
        } else if let Some(headers) = &self.cached_sps_pps {
            let mut output = Vec::with_capacity(headers.len() + packet.len());
            output.extend_from_slice(headers);
            output.extend_from_slice(packet);
            Ok(output)
        } else {
            bail!("NVENC produced a frame without initial SPS/PPS");
        }
    }

    pub fn force_idr(&mut self) {
        self.force_idr = true;
    }
}

fn error_message(error: &[c_char; 512]) -> String {
    // SAFETY: The zero-initialized buffer is written with snprintf by the C bridge.
    unsafe { CStr::from_ptr(error.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn validate_input(width: usize, height: usize, length: usize, stride: usize) -> Result<()> {
    ensure!(width > 0 && height > 0, "empty NVENC input");
    let row = width.checked_mul(4).context("NVENC row size overflow")?;
    ensure!(
        stride >= row && stride <= i32::MAX as usize,
        "invalid NVENC BGRA stride"
    );
    let required = stride
        .checked_mul(height - 1)
        .and_then(|n| n.checked_add(row))
        .context("NVENC input size overflow")?;
    ensure!(length >= required, "NVENC BGRA buffer is too short");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use openh264::formats::YUVSource;

    #[test]
    fn validates_padded_rows_and_truncated_input_without_a_gpu() {
        assert!(validate_input(4, 2, 36, 20).is_ok());
        assert!(validate_input(4, 2, 35, 20).is_err());
        assert!(validate_input(4, 2, 32, 15).is_err());
        assert!(validate_input(4, 0, 0, 16).is_err());
        assert!(validate_input(4, 2, usize::MAX, usize::MAX).is_err());
        assert!(validate_input(usize::MAX, 2, 32, 16).is_err());
    }

    #[test]
    fn rejects_invalid_encoder_configuration_before_opening_gpu() {
        for (w, h, fps, quality) in [
            (0, 64, 30, 23),
            (63, 64, 30, 23),
            (64, 63, 30, 23),
            (64, 64, 0, 23),
            (64, 64, 30, 52),
        ] {
            assert!(
                NvencEncoder::new(w, h, 1_000_000, fps, quality, H264RateControl::Vbr).is_err()
            );
        }
    }

    #[test]
    #[ignore = "requires an NVIDIA GPU, NVENC driver and FFmpeg h264_nvenc"]
    fn nvenc_hardware_roundtrip_and_forced_idr() {
        for (width, height) in [(320, 240), (1920, 1080)] {
            for mode in [H264RateControl::Vbr, H264RateControl::Cqp] {
                let mut enc = NvencEncoder::new(width, height, 10_000_000, 30, 23, mode).unwrap();
                let stride = width as usize * 4 + 32;
                let mut pixels = vec![0; stride * height as usize];
                for row in pixels.chunks_mut(stride) {
                    for pixel in row[..width as usize * 4].as_chunks_mut::<4>().0 {
                        pixel.copy_from_slice(&[0, 0, 255, 255]);
                    }
                }
                let mut decoder = openh264::decoder::Decoder::new().unwrap();
                for index in 0..5 {
                    if index == 3 {
                        enc.force_idr();
                    }
                    let packet = enc.encode(&pixels, stride).unwrap();
                    assert!(!packet.is_empty());
                    let is_idr = packet
                        .windows(4)
                        .any(|w| w[..3] == [0, 0, 1] && w[3] & 31 == 5);
                    assert_eq!(is_idr, index == 0 || index == 3);
                    if index == 3 {
                        // A newly connected decoder must recover from this packet alone.
                        decoder = openh264::decoder::Decoder::new().unwrap();
                    }
                    let decoded = decoder
                        .decode(&packet)
                        .unwrap()
                        .expect("immediate decoded frame");
                    assert_eq!(decoded.dimensions(), (width as usize, height as usize));
                    let mut rgb = vec![0; width as usize * height as usize * 3];
                    decoded.write_rgb8(&mut rgb);
                    let p = &rgb[(height as usize / 2 * width as usize + width as usize / 2) * 3..]
                        [..3];
                    assert!(
                        p[0] > 200 && p[1] < 60 && p[2] < 60,
                        "red BGRA source decoded as {p:?}"
                    );
                }
            }
        }
    }
}
