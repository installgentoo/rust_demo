#![warn(clippy::all)]
#![allow(clippy::range_plus_one, clippy::many_single_char_names, clippy::too_many_arguments, clippy::cast_lossless)]

use grafix_toolbox::{gui::*, uses::math::*, uses::*, GL::mesh::*, GL::pbrt::*, GL::window::*, GL::*, *};

fn main() {
	LOGGER!(logging::Term, INFO);

	ShaderManager::LoadSources("shd_support.glsl");
	ShaderManager::LoadSources("shd_pbrt.glsl");

	let mut window = TIMER!(window, {
		let win = EXPECT!(Window::get((50, 50, 1600, 900), "Engine"));
		GLEnable!(DEPTH_TEST, BLEND, MULTISAMPLE, TEXTURE_CUBE_MAP_SEAMLESS, DEPTH_WRITEMASK);
		GLDisable!(CULL_FACE);
		GL::BlendFunc::Set((gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA));
		GL::DepthFunc::Set(gl::LESS);
		GL::EnableDebugContext(GL::DebugLevel::All);
		win
	});

	let font = {
		let alphabet = (9..=9).chain(32_u8..127).map(|n| n as char).collect::<String>() + "ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ";
		Font::new_cached("UbuntuMono-R", &alphabet)
	};

	let mut renderer = Renderer::new(Theme {
		easing: 10.,
		bg: (0.2, 0.2, 0.2, 0.7),
		bg_focus: hex_to_rgba(0x596475A0),
		fg: hex_to_rgba(0x626975FF),
		fg_focus: hex_to_rgba(0x461E5CCF),
		highlight: (0.9, 0.4, 0.1, 1.),
		text: (1., 0.9, 0.9, 0.9),
		text_focus: (1., 1., 1., 1.),
		text_highlight: (0.2, 0.2, 0.2, 1.),
		font,
		font_size: 0.8,
	});

	let atlas = TexAtlas::<RGBA>::new();
	let mut spinner = Animation::from_file("spinner", &atlas);

	let (skybox, brdf_lut) = (Environment::new_cached("lythwood_lounge_4k"), Environment::lut_cached());

	let (mut skybox_shd, mut render_shd) = (
		EXPECT!(Shader::new(("vs_skybox", "ps_skybox"))),
		EXPECT!(Shader::new(("vs_material_based_render", "ps_material_based_render"))),
	);

	let (sampl, mipmapped) = (
		&Sampler::linear(),
		&Sampler!(
			(gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE),
			(gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE),
			(gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE),
			(gl::TEXTURE_MIN_FILTER, gl::LINEAR_MIPMAP_LINEAR)
		),
	);
	let mut sphere = Mesh::make_sphere(0.1, 64);
	let mut dragon = Mesh::new_cached("stanford_dragon");

	let large_text: String = (0..100)
		.map(|_| "Functional textbox is capable of showing millions of lines with negligeble cpu usage!(so long as we're not editing, but that can be solved functionally as well)\nBiggest issue is collecting the text to display, really - you can set large text range to 1000000 in release.\n\nUpper bar allows to drag layout around, pip on the bottom right resizes it.\n\n\n")
		.collect();

	Layout::storage(ID!("menu")).pos = (-10., -10.);
	Layout::storage(ID!("menu")).size = (1.5, 2.);
	TextEdit::storage(ID!("text")).text = large_text.into();
	Slider::storage(ID!("metallicity")).pip_pos = 0.885;
	Slider::storage(ID!("roughness")).pip_pos = 0.286;
	Slider::storage(ID!("rotation")).pip_pos = 0.1;

	let show_tooltip = |hovered, p: Vec2, msg: &str, r: &mut RenderLock| {
		if timeout(hovered) {
			let size = (0.03, 0.1).mul((msg.len(), 1));
			r.clip(p, size);
			r.Label(ID!("Tooltip"), p.sum(size.mul(0.05)), size, msg);
			r.unclip();
		}
	};

	let mut exit = false;
	let mut mouse_pos = (0., 0.);
	let mut magnification = 0.3;
	let mut rotation = 0_f32;

	for i in (0..100).chain((0..100).step_by(1).rev()).map(|i| f32::to(i) / 100.).cycle() {
		let a = Window::aspect();
		let mut cam1 = Camera::new(
			glm::perspective(a.y() / a.x(), 70_f32.to_radians(), 0.1, 10.),
			glm::look_at(&glm::vec3(0., 0., -5.), &glm::vec3(0., 0., 0.), &glm::vec3(0., 1., 0.)),
		);

		let mut r = renderer.lock();

		{
			window.draw_to_screen();
			let metallicity = 0.995 * r.Slider(ID!("metallicity"), (0.3, -0.88), (1., 0.05), 0.05);
			let roughness = (0.05 + r.Slider(ID!("roughness"), (0.3, -0.94), (1., 0.05), 0.05)) / 1.05;
			r.Label(ID!("metal_v"), (1.31, -0.88), (0.3, 0.05), &format!("Metal: {:.3}", metallicity));
			r.Label(ID!("rough_v"), (1.31, -0.94), (0.3, 0.05), &format!("Rough: {:.3}", roughness));

			let t = 1. - i;

			use glm::{vec3 as v3, vec4 as v4, Mat4 as m4};
			let m = magnification;
			GL::ClearScreen((0., 1.));
			let model = &glm::translate(
				&glm::rotate(&glm::scale(&glm::identity(), &v3(m, m, m)), 90_f32.to_radians(), &v3(-1., 0., 0.)),
				&v3(0., 0., -0.3),
			);
			let (c, s, cam_world) = {
				rotation += r.Slider(ID!("rotation"), (0.3, -1.), (1., 0.05), 0.05) * 0.01;
				let (c, s) = (rotation.cos(), rotation.sin()).mul(0.3);
				let t = t.to_radians() * 90.;
				(t.cos(), t.sin(), v3(s, 0., c))
			};

			cam1.setView(glm::look_at(&cam_world, &v3(0., 0., 0.), &v3(0., 1., 0.)));
			let view = m4::to(cam1.V());
			let (irr, spec, lut) = (skybox.irradiance.Bind(sampl), skybox.specular.Bind(mipmapped), brdf_lut.Bind(sampl));
			let _ = Uniforms!(
				render_shd,
				("irradiance_cubetex", &irr),
				("specular_cubetex", &spec),
				("brdf_lut", &lut),
				("MVPMat", cam1.MVP(model)),
				("ModelViewMat", cam1.MV(model)),
				("NormalViewMat", cam1.NV(model)),
				("NormalMat", cam1.N(model)),
				("camera_world", Vec3::to(cam_world)),
				(
					"light_pos",
					&[
						Vec4::to(view * v4(6. * c, 6. * s, 0., 1.)).xyz(),
						Vec4::to(view * v4(2. * c, 0., 2. * s, 1.)).xyz(),
						Vec4::to(view * v4(c, s, -2., 1.)).xyz(),
						Vec4::to(view * v4(-1.5 * s, 1. * c, -2., 1.)).xyz()
					][..]
				),
				("light_color", &[(1., 0., 0., 2.), (0., 0., 1., 2.), (1., 0., 1., 2.), (0., 1., 0., 2.)][..]),
				("albedo", hex_to_rgba(0xD4AF3700).xyz()),
				("metallicity", metallicity),
				("roughness", roughness),
				("exposure", 1.),
				("max_lod", skybox.mip_levels)
			);
			dragon.Draw();
			sphere.Draw();
		}
		{
			let s = skybox.specular.Bind(mipmapped);
			let _ = Uniforms!(skybox_shd, ("skybox_tex", &s), ("MVPMat", cam1.VP()), ("exposure", 1.));
			Skybox::Draw();
		}

		r.Layout(ID!("menu"), |r, (pos, size)| {
			let (button_w, button_h, padding) = (0.18, 0.06, 0.01);
			let button = |n| {
				(
					pos.sum(size.mul((button_w * f32::to(n) + padding * f32::to(n * 2 + 1), padding))),
					size.mul((button_w, button_h)),
				)
			};

			r.TextEdit(
				ID!("text"),
				pos.sum(size.mul((0, button_h + padding * 2.))),
				size.mul((1, 1. - (button_h + padding * 3.))),
				0.05,
			);

			let (p, s) = button(4);
			let pressed = r.Button(ID!("exit"), p, s, "Exit");
			exit |= pressed;
			show_tooltip(r.hovers_in((p, p.sum(s))) && !pressed, p.sum(s.mul(0.5)), "GUI is easy!", r);
		});

		r.draw(primitives::Sprite {
			pos: mouse_pos,
			size: (40., 40.).div(Window::aspect()).div(Window::size()),
			color: (1., 1. - i, 1., 1.),
			tex: spinner.frame(i),
		});

		let mut events = window.poll_events();
		events.iter().for_each(|e| {
			if let MouseMove { at, .. } = e {
				mouse_pos = *at;
			}
		});
		renderer = r.unlock(&mut events);
		window.swap();

		for e in events {
			match e {
				Keyboard { key, state } if Key::Escape == key && state.pressed() => exit = true,
				Scroll { at, .. } => {
					magnification = (magnification + 0.01 * at.y()).clamp(0.01, 1.);
					rotation += 0.2 * at.x()
				}
				_ => (), //println!("{:?}", e),
			}
		}

		if exit {
			return;
		}
		use events::{Event::*, *};
	}
}

fn timeout(active: bool) -> bool {
	unsafe {
		static mut TIME: usize = 0;
		TIME += 1;
		if !active {
			TIME = 0
		}
		TIME > 60
	}
}
