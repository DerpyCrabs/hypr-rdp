fn main() {
    println!("cargo:rerun-if-changed=src/egfx/nvenc_shim.c");
    #[cfg(feature = "nvenc")]
    {
        let mut build = cc::Build::new();
        for library in ["libavcodec", "libavutil", "libswscale"] {
            let found = pkg_config::Config::new()
                .probe(library)
                .unwrap_or_else(|error| {
                    panic!("nvenc requires FFmpeg development libraries: {error}")
                });
            build.includes(found.include_paths);
        }
        build
            .file("src/egfx/nvenc_shim.c")
            .warnings(true)
            .compile("hypr_nvenc");
    }
}
