use std::process::Command;
fn main() {
	Command::new("./link_shaders.sh").spawn().expect("couldn't link .glsl files");
	Command::new("ln").arg("-rfs").arg("res").arg("target").spawn().expect("couldn't link resources");
}
